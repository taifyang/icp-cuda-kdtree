#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "memory.h"
#include "icp.h"
#include "svd.h"
#include "cukd/builder.h"
#include "cukd/fcp.h"


namespace
{
    // One thread per correspondence: the traversal is latency bound and
    // diverges heavily between warps, so small blocks win on load balance.
    constexpr int nn_threads = 32;
    constexpr int fit_threads = 256;
    // Caps the number of partial results, and with it the reduction we do on
    // the host and the size of the transfer.
    constexpr int max_fit_blocks = 2048;
    // Per block: the 9 covariance terms, the residual sum and the match count.
    constexpr int partial_width = 11;

    struct Rigid
    {
        float r[9]; // row major
        float t[3];
    };

    Rigid to_rigid(const Eigen::Matrix3f& rotation, const Eigen::Vector3f& translation)
    {
        Rigid rigid;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                rigid.r[i * 3 + j] = rotation(i, j);
        rigid.t[0] = translation.x();
        rigid.t[1] = translation.y();
        rigid.t[2] = translation.z();
        return rigid;
    }

    float3 to_float3(const Eigen::Vector3f& v)
    {
        return make_float3(v.x(), v.y(), v.z());
    }

    Eigen::Vector3f centroid(const std::vector<Eigen::Vector3f>& points)
    {
        Eigen::Matrix<double, 3, 1> sum = Eigen::Matrix<double, 3, 1>::Zero();
        for (const auto& point : points)
            sum += point.cast<double>();
        return (sum / static_cast<double>(points.size())).cast<float>();
    }

    template <typename T>
    __device__ __forceinline__ T warp_sum(T value)
    {
    #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        return value;
    }

    __global__ void center_kernel(float3* points, int num_points, float3 center)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= num_points)
            return;

        float3 point = points[i];
        points[i] = make_float3(point.x - center.x, point.y - center.y, point.z - center.z);
    }

    __global__ void apply_alignment_kernel(const float3* source, float3* destination, int num_points, Rigid transform)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= num_points)
            return;

        float3 p = source[i];
        destination[i] = make_float3(
            transform.r[0] * p.x + transform.r[1] * p.y + transform.r[2] * p.z + transform.t[0],
            transform.r[3] * p.x + transform.r[4] * p.y + transform.r[5] * p.z + transform.t[1],
            transform.r[6] * p.x + transform.r[7] * p.y + transform.r[8] * p.z + transform.t[2]);
    }

    __global__ void d_fcp(float3* results, const float3* queries, int num_queries, float3 query_center,
        const cukd::box_t<float3>* bounds, const float3* nodes, int num_nodes, float cut_off_radius)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= num_queries)
            return;

        float3 query = queries[tid];
        float3 query_pos = make_float3(query.x - query_center.x, query.y - query_center.y, query.z - query_center.z);

        cukd::FcpSearchParams params;
        params.cutOffRadius = cut_off_radius;
        int closest_id = cukd::cct::fcp(query_pos, *bounds, nodes, num_nodes, params);

        results[tid] = closest_id < 0 ? make_float3(NAN, NAN, NAN) : nodes[closest_id];
    }

    /**
        * @brief Accumulate the covariance of the (centered) source against its
        * correspondences, and the residual, in a single pass over the clouds.
        * Each block writes one row of partial sums; the caller reduces them.
        */
    __global__ void fit_kernel(const float3* points, const float3* correspondences, int num_points,
        float3 point_center, float3 correspondence_center, float* partials)
    {
        constexpr int warps = fit_threads / 32;
        __shared__ float warp_covariance[warps][9];
        __shared__ float warp_residual[warps];
        __shared__ int warp_matches[warps];

        float covariance[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float residual = 0.f;
        int matches = 0;

        const int stride = blockDim.x * gridDim.x;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_points; i += stride)
        {
            const float3 y = correspondences[i];
            if (isnan(y.x)) // rejected by the cut off radius
                continue;

            const float3 p = points[i];
            const float px = p.x - point_center.x;
            const float py = p.y - point_center.y;
            const float pz = p.z - point_center.z;

            covariance[0] += px * y.x; covariance[1] += px * y.y; covariance[2] += px * y.z;
            covariance[3] += py * y.x; covariance[4] += py * y.y; covariance[5] += py * y.z;
            covariance[6] += pz * y.x; covariance[7] += pz * y.y; covariance[8] += pz * y.z;

            const float dx = y.x + correspondence_center.x - p.x;
            const float dy = y.y + correspondence_center.y - p.y;
            const float dz = y.z + correspondence_center.z - p.z;
            residual += sqrtf(dx * dx + dy * dy + dz * dz);
            ++matches;
        }

    #pragma unroll
        for (int k = 0; k < 9; ++k)
        {
            float total = warp_sum(covariance[k]);
            if ((threadIdx.x & 31) == 0)
                warp_covariance[threadIdx.x >> 5][k] = total;
        }
        float total_residual = warp_sum(residual);
        int total_matches = warp_sum(matches);
        if ((threadIdx.x & 31) == 0)
        {
            warp_residual[threadIdx.x >> 5] = total_residual;
            warp_matches[threadIdx.x >> 5] = total_matches;
        }
        __syncthreads();

        // The per-warp partials are combined in a fixed order so that the
        // result does not depend on the scheduling of the block's warps.
        const int active_warps = (blockDim.x + 31) >> 5;
        if (threadIdx.x < 9)
        {
            float total = 0.f;
            for (int w = 0; w < active_warps; ++w)
                total += warp_covariance[w][threadIdx.x];
            partials[blockIdx.x * partial_width + threadIdx.x] = total;
        }
        else if (threadIdx.x == 9)
        {
            float total = 0.f;
            for (int w = 0; w < active_warps; ++w)
                total += warp_residual[w];
            partials[blockIdx.x * partial_width + 9] = total;
        }
        else if (threadIdx.x == 10)
        {
            int total = 0;
            for (int w = 0; w < active_warps; ++w)
                total += warp_matches[w];
            partials[blockIdx.x * partial_width + 10] = static_cast<float>(total);
        }
    }
} // namespace


void icp_warmup()
{
    float* probe = cuda::mallocRaw<float>(1);
    apply_alignment_kernel<<<1, 1>>>(reinterpret_cast<float3*>(probe), reinterpret_cast<float3*>(probe), 0, Rigid());
    cudaDeviceSynchronize();
    cuda::free(probe);
}

std::tuple<Eigen::Matrix4f, std::vector<Eigen::Vector3f>>
icp(const std::vector<Eigen::Vector3f>& m, const std::vector<Eigen::Vector3f>& p, size_t iterations, float threshold,
    float cut_off_radius, float relative_threshold)
{
    const int num_reference = static_cast<int>(m.size());
    const int num_source = static_cast<int>(p.size());
    if (num_reference == 0 || num_source == 0)
        return {Eigen::Matrix4f::Identity(), p};

    const Eigen::Vector3f mu_m = centroid(m);
    Eigen::Vector3f mu_p = centroid(p);

    cuda::ptr_t<float3> d_tree = cuda::malloc<float3>(num_reference);
    cuda::ptr_t<float3> d_source = cuda::malloc<float3>(num_source);
    cuda::ptr_t<float3> d_aligned = cuda::malloc<float3>(num_source);
    cuda::ptr_t<float3> d_correspondences = cuda::malloc<float3>(num_source);
    cuda::ptr_t<cukd::box_t<float3>> d_bounds = cuda::malloc<cukd::box_t<float3>>(1);
    const int fit_blocks = std::min((num_source + fit_threads - 1) / fit_threads, max_fit_blocks);
    cuda::ptr_t<float> d_partials = cuda::malloc<float>(fit_blocks * partial_width);

    const float3 mu_m_3f = to_float3(mu_m);
    cudaMemcpy(d_tree.get(), m.data(), num_reference * sizeof(float3), cudaMemcpyHostToDevice);
    center_kernel<<<(num_reference + 255) / 256, 256>>>(d_tree.get(), num_reference, mu_m_3f);
    cudaMemcpy(d_source.get(), p.data(), num_source * sizeof(float3), cudaMemcpyHostToDevice);

    cukd::buildTree(d_tree.get(), num_reference, d_bounds.get());

    std::vector<float> partials(static_cast<size_t>(fit_blocks) * partial_width);

    Eigen::Matrix3f cumulative_rotation = Eigen::Matrix3f::Identity();
    Eigen::Vector3f cumulative_translation = Eigen::Vector3f::Zero();
    Eigen::Matrix3f best_rotation = cumulative_rotation;
    Eigen::Vector3f best_translation = cumulative_translation;

    float previous_error = std::numeric_limits<float>::infinity();
    float best_error = previous_error;

    for (size_t i = 0; i < iterations; ++i)
    {
        Rigid transform = to_rigid(cumulative_rotation, cumulative_translation);
        apply_alignment_kernel<<<(num_source + 255) / 256, 256>>>(d_source.get(), d_aligned.get(), num_source, transform);

        d_fcp<<<(num_source + nn_threads - 1) / nn_threads, nn_threads>>>(
            d_correspondences.get(), d_aligned.get(), num_source, to_float3(mu_p), d_bounds.get(), d_tree.get(), num_reference, cut_off_radius);

        fit_kernel<<<fit_blocks, fit_threads>>>(d_aligned.get(), d_correspondences.get(), num_source, to_float3(mu_p), mu_m_3f, d_partials.get());

        cudaMemcpy(partials.data(), d_partials.get(), partials.size() * sizeof(float), cudaMemcpyDeviceToHost);

        double sums[10] = {0.0};
        double matches = 0.0;
        for (int block = 0; block < fit_blocks; ++block)
        {
            const float* row = partials.data() + static_cast<size_t>(block) * partial_width;
            for (int k = 0; k < 9; ++k)
                sums[k] += row[k];
            sums[9] += row[9];
            matches += row[10];
        }
        if (matches == 0.0) // the cut off radius rejected every correspondence
            break;

        const float error = static_cast<float>(sums[9] / matches);
        if (error < best_error)
        {
            best_error = error;
            best_rotation = cumulative_rotation;
            best_translation = cumulative_translation;
        }

        //std::cout << "iter: " << (i + 1) << "/" << iterations << std::endl;
        //std::cout << "error: " << error << std::endl;

        if (error <= threshold)
            break;
        if (previous_error - error < relative_threshold * std::max(error, FLT_MIN))
            break;
        previous_error = error;

        Eigen::Matrix3f covariance;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                covariance(r, c) = static_cast<float>(sums[r * 3 + c]);

        const Eigen::Matrix3f step_rotation = find_rotation(covariance);
        const Eigen::Vector3f step_translation = mu_m - step_rotation * mu_p;

        cumulative_translation = step_rotation * cumulative_translation + step_translation;
        cumulative_rotation = step_rotation * cumulative_rotation;
        // The Kabsch step always maps the source centroid onto the target one,
        // so the mean never has to be reduced from the cloud again.
        mu_p = mu_m;
    }

    apply_alignment_kernel<<<(num_source + 255) / 256, 256>>>(
        d_source.get(), d_aligned.get(), num_source, to_rigid(best_rotation, best_translation));

    std::vector<Eigen::Vector3f> result(num_source);
    cudaMemcpy(result.data(), d_aligned.get(), num_source * sizeof(float3), cudaMemcpyDeviceToHost);

    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    transformation.block<3, 3>(0, 0) = best_rotation;
    transformation.block<3, 1>(0, 3) = best_translation;

    return {transformation, result};
}
