#pragma once

#include "point3d.h"


/**
    * @brief The ICP algorithm on GPU.
    * @param m The reference point cloud.
    * @param p The transformed point cloud.
    * @param iterations The maximum number of iterations.
    * @param threshold The error threshold.
    * @return The transformation matrix and the p transformed.
    */
std::tuple<Eigen::Matrix4f, std::vector<Eigen::Vector3f>>
icp(const std::vector<Eigen::Vector3f>& m, const std::vector<Eigen::Vector3f>& p, size_t iterations, float threshold);
