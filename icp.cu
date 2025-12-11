#include <limits>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "memory.h"
#include "icp.h"
#include "matrix.h"
#include "svd.h"
#include "cukd/builder.h"
#include "cukd/fcp.h"


__global__ void to_transformation_kernel(GPUMatrix rotation, GPUMatrix translation, GPUMatrix res)
{
    int i = threadIdx.x;
    int j = threadIdx.y;

    if (i < 3)
    {
        if (j < 3)
            res(i, j) = rotation(i, j);
        else // i < 3 && j == 3
            res(i, 3) = translation(0, i);
    }
    else // i == 3
    {
        if (j < 3)
            res(3, j) = 0;
        else // i == 3 && j == 3
            res(3, 3) = 1;
    }
}

GPUMatrix to_transformation(const GPUMatrix& rotation, const GPUMatrix& translation)
{
    assert(rotation.rows == 3);
    assert(rotation.cols == 3);
    assert(translation.rows == 1);
    assert(translation.cols == 3);

    GPUMatrix transformation(4, 4);

    dim3 blockdim(4, 4);
    to_transformation_kernel<<<1, blockdim>>>(rotation, translation, transformation);

    return transformation;
}

GPUMatrix find_alignment(const GPUMatrix& p_centered, const GPUMatrix& y, const GPUMatrix& mu_p, const GPUMatrix& mu_m)
{
    auto covariance = GPUMatrix::find_covariance(p_centered, y);
    auto rotation = GPUMatrix::from_cpu(find_rotation(covariance.to_cpu()));
    auto translation = mu_m.subtract(mu_p.dot(rotation.transpose()));

    return to_transformation(rotation, translation);
}

__global__ void compute_error_kernel(GPUMatrix m, GPUMatrix p, GPUMatrix mu_m, GPUMatrix diffs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m.rows)
        return;

    assert(m.cols == p.cols);
    assert(m.cols == mu_m.cols);
    assert(mu_m.rows == 1);

    float dist = 0;
    for (size_t j = 0; j < m.cols; ++j)
    {
        float diff = m(i, j) + mu_m(0, j) - p(i, j);
        dist += diff * diff;
    }

    diffs(i, 0) = sqrt(dist);
}

float compute_error(const GPUMatrix& m, const GPUMatrix& p, const GPUMatrix& mu_m)
{
    GPUMatrix diffs(m.rows, 1);

    dim3 blockdim(1024);
    dim3 griddim((m.rows + blockdim.x - 1) / blockdim.x);
    compute_error_kernel<<<griddim, blockdim>>>(m, p, mu_m, diffs);

    return diffs.sum_colwise().to_cpu()(0, 0) / m.rows;
}

__global__ void apply_alignment_kernel(GPUMatrix p, GPUMatrix transformation)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= p.rows)
        return;

    float values[3] = {0};
    for (size_t j = 0; j < p.cols; ++j)
    {
        for (size_t k = 0; k < 3; ++k)
            values[j] += p(i, k) * transformation(j, k);
        values[j] += transformation(j, 3);
    }

    for (size_t j = 0; j < p.cols; ++j)
        p(i, j) = values[j];
}

void apply_alignment(GPUMatrix& p, const GPUMatrix& transformation)
{
    assert(p.cols == 3);
    assert(transformation.rows == 4);
    assert(transformation.cols == 4);

    dim3 blockdim(1024);
    dim3 griddim((p.rows + blockdim.x - 1) / blockdim.x);
    apply_alignment_kernel<<<griddim, blockdim>>>(p, transformation);
}

__global__ void d_fcp(float3   *d_results,
        float3  *d_queries,
        int      numQueries,
        const cukd::box_t<float3> *d_bounds,
        float3  *d_nodes,
        int      numNodes,
        float    cutOffRadius)
{
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numQueries) return;

    float3 queryPos = d_queries[tid];
    cukd::FcpSearchParams params;
    params.cutOffRadius = cutOffRadius;
    int closestID = cukd::cct::fcp(queryPos,*d_bounds,d_nodes,numNodes,params);    
    d_results[tid]= d_nodes[closestID];
}

std::tuple<Eigen::Matrix4f, std::vector<Eigen::Vector3f>>
icp(const std::vector<Eigen::Vector3f>& q, const std::vector<Eigen::Vector3f>& p, size_t iterations, float threshold)
{
    auto new_p = GPUMatrix::from_vector(p);
    auto m = GPUMatrix::from_vector(q);
    auto mu_m = m.mean();
    auto transformation = GPUMatrix::Identity(4);
    float error = std::numeric_limits<float>::infinity();
    float3* d_points = m.subtract_rowwise(mu_m).to_float3(q.size());

    cukd::box_t<float3> *d_bounds;
    cudaMallocManaged((void**)&d_bounds, sizeof(cukd::box_t<float3>));

    cukd::buildTree(d_points, q.size(), d_bounds);

    float3 *d_results;
    cudaMallocManaged((void**)&d_results, q.size()*sizeof(*d_results));
    float3 *d_queries;
    cudaMallocManaged((char **)&d_queries, q.size()*sizeof(*d_queries));
    GPUMatrix y(p.size(), 3);

    for (size_t i = 0; i < iterations && error > threshold; ++i)
    {
        auto mu_p = new_p.mean();
        auto p_centered = new_p.subtract_rowwise(mu_p);
        d_queries = (float3*)p_centered.ptr;
        
        d_fcp<<<p.size(), 1>>> (d_results, d_queries, p.size(), d_bounds, d_points, q.size(), std::numeric_limits<float>::infinity());
        cudaDeviceSynchronize();

        y.ptr = (float*)d_results;
        auto new_transformation = find_alignment(p_centered, y, mu_p, mu_m);

        transformation = new_transformation.dot(transformation);
        apply_alignment(new_p, new_transformation);
        error = compute_error(y, new_p, mu_m);

//#ifdef _DEBUG
        std::cout << "iter: " << (i + 1) << "/" << iterations << std::endl;
        std::cout << "error: " << error << std::endl;          
//#endif
    }
    return {transformation.to_cpu().transpose(), new_p.to_vector()};
}

