#include "point3d.h"


std::vector<Eigen::Vector3f> read_vector(std::vector<Eigen::Vector3f>& pts)
{
    std::vector<Eigen::Vector3f> points(pts);
    return points;
}


float squared_distance(const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{
    return (a - b).squaredNorm();
}

size_t closest(const Eigen::Vector3f& a, const std::vector<Eigen::Vector3f>& v)
{
    assert(v.size() > 0);

    size_t ret = 0;
    float dist = squared_distance(a, v[ret]);

    for (size_t i = 1; i < v.size(); i++)
    {
        float tmp_dist = squared_distance(a, v[i]);
        if (tmp_dist < dist)
        {
            dist = tmp_dist;
            ret = i;
        }
    }

    return ret;
}

std::vector<Eigen::Vector3f> closest(const std::vector<Eigen::Vector3f>& a, const std::vector<Eigen::Vector3f>& b)
{
    std::vector<Eigen::Vector3f> v(a.size());

//#pragma omp parallel for
    for (size_t i = 0; i < a.size(); ++i)
        v[i] = b[closest(a[i], b)];

    return v;
}

Eigen::Vector3f mean(const std::vector<Eigen::Vector3f>& a)
{
    size_t len = a.size();
    float x = 0;
    float y = 0;
    float z = 0;

    for (size_t i = 0; i < len; ++i)
    {
        x += a[i].x() / len;
        y += a[i].y() / len;
        z += a[i].z() / len;
    }

    return Eigen::Vector3f{x, y, z};
}

std::vector<Eigen::Vector3f> subtract(const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3f& mean)
{
    std::vector<Eigen::Vector3f> centered;
    centered.resize(points.size());

//#pragma omp parallel for
    for (size_t i = 0; i < points.size(); ++i)
    {
        centered[i] = Eigen::Vector3f{points[i].x() - mean.x(),points[i].y() - mean.y(),points[i].z() - mean.z()};
    }

    return centered;
}

std::tuple<float, float, float, float, float, float, float, float, float>
find_covariance(const std::vector<Eigen::Vector3f>& p_centered, const std::vector<Eigen::Vector3f>& y_centered)
{
    float sxx = 0, sxy = 0, sxz = 0, syx = 0, syy = 0, syz = 0, szx = 0, szy = 0, szz = 0;

    for (size_t i = 0; i < p_centered.size(); ++i)
    {
#define ADDPROD(FIRST_COORD, SECOND_COORD)                                     \
s##FIRST_COORD##SECOND_COORD +=                                            \
    (p_centered[i].FIRST_COORD()) * (y_centered[i].SECOND_COORD())
        ADDPROD(x, x);
        ADDPROD(x, y);
        ADDPROD(x, z);
        ADDPROD(y, x);
        ADDPROD(y, y);
        ADDPROD(y, z);
        ADDPROD(z, x);
        ADDPROD(z, y);
        ADDPROD(z, z);
#undef ADDPROD
    }

    return {sxx, sxy, sxz, syx, syy, syz, szx, szy, szz};
}

float sum_of_squared_norms(const std::vector<Eigen::Vector3f>& a)
{
    float sum = 0.0;

    for (const auto& value : a)
        sum += value.squaredNorm();

    return sum;
}

