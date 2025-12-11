#pragma once

#include <ostream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <omp.h>


std::vector<Eigen::Vector3f> read_vector(std::vector<Eigen::Vector3f>& pts);
/**
    * @brief Compute the squared distance between two points.
    * @param a The first point.
    * @param b The second point.
    * @return The squared distance.
    */
float squared_distance(const Eigen::Vector3f& a, const Eigen::Vector3f& b);

/**
    * @brief Find the closest point to a point in a point list.
    * @param a The query.
    * @param v The points.
    * @return The index of the closest point in the points.
    */
size_t closest(const Eigen::Vector3f& a, const std::vector<Eigen::Vector3f>& v);

/**
    * @brief Find the closest points in the second point list to each point in
    * the first point list.
    * @param a The queries.
    * @param b The points.
    * @return The closest points.
    */
std::vector<Eigen::Vector3f> closest(const std::vector<Eigen::Vector3f>& a, const std::vector<Eigen::Vector3f>& b);

/**
    * @brief Compute the mean of a point list.
    * @param a The point list.
    * @return The mean.
    */
Eigen::Vector3f mean(const std::vector<Eigen::Vector3f>& a);

/**
    * @brief Compute the sum of squared norms of a point list.
    * @param a The point list.
    * @return The sum of squared norms.
    */
float sum_of_squared_norms(const std::vector<Eigen::Vector3f>& a);

/**
    * @brief Subtract a vector to each point in a point list.
    * @param points The point list.
    * @param mean The vector.
    * @return The point list containing the subtraction.
    */
std::vector<Eigen::Vector3f> subtract(const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3f& mean);

/**
    * @brief Compute the covariance matrix between two point clouds.
    * @param p_centered The first point cloud.
    * @param y_centered The second point cloud.
    * @return The covariance matrix.
    */
std::tuple<float, float, float, float, float, float, float, float, float>
find_covariance(const std::vector<Eigen::Vector3f>& p_centered, const std::vector<Eigen::Vector3f>& y_centered);

