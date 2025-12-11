#pragma once

#include <Eigen/Dense>


/**
    * @brief Determine the rotation matrix using a SVD with eigen.
    * @param covariance The covariance matrix.
    * @return The rotation matrix.
    */
Eigen::Matrix3f find_rotation(const Eigen::Matrix3f& covariance);
