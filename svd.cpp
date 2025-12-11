#include "svd.h"


Eigen::Matrix3f find_rotation(const Eigen::Matrix3f& covariance)
{
	Eigen::Matrix3f matrix = covariance;
	Eigen::JacobiSVD svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3f rotation = svd.matrixU() * svd.matrixV().transpose();
	return rotation.transpose();
}
