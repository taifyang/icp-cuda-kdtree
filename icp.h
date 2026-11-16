#pragma once

#include <limits>

#include "point3d.h"


/**
    * @brief Force the CUDA context and the module holding the ICP kernels to
    * be loaded, so that this one-off cost is not attributed to icp().
    */
void icp_warmup();

/**
    * @brief The ICP algorithm on GPU.
    * @param m The reference point cloud.
    * @param p The transformed point cloud.
    * @param iterations The maximum number of iterations.
    * @param threshold Stop once the mean distance drops below this.
    * @param cut_off_radius Correspondences further than this are rejected.
    * @param relative_threshold Stop once an iteration improves the mean
    * distance by less than this fraction. The best iterate, not the last one,
    * is returned: point-to-point ICP on a dense cloud starts drifting once the
    * correspondences degenerate, which makes the residual grow again.
    * @return The transformation matrix and the p transformed.
    */
std::tuple<Eigen::Matrix4f, std::vector<Eigen::Vector3f>>
icp(const std::vector<Eigen::Vector3f>& m, const std::vector<Eigen::Vector3f>& p, size_t iterations, float threshold,
    float cut_off_radius = std::numeric_limits<float>::infinity(), float relative_threshold = 1e-5f);
