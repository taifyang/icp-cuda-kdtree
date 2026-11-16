#include <iostream>
#include <fstream>
#include <chrono>
#include "icp.h"


int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cerr << "usage: " << argv[0] << " <source cloud> <target cloud>" << std::endl;
		return 1;
	}

	std::fstream cloud_src, cloud_tgt;
	cloud_src.open(argv[1]);
	cloud_tgt.open(argv[2]);

	if (!cloud_src.is_open())
	{
		std::cerr << "cannot open " << argv[1] << std::endl;
		return 1;
	}
	if (!cloud_tgt.is_open())
	{
		std::cerr << "cannot open " << argv[2] << std::endl;
		return 1;
	}

	std::vector<Eigen::Vector3f> vector_src, vector_tgt;

	float x, y, z, r, g, b;
	while (cloud_src >> x >> y >> z >>r>>g>>b)
	{
		vector_src.push_back(Eigen::Vector3f(x, y, z));
	}
	while (cloud_tgt >> x >> y >> z>>r>>g>>b)
	{
		vector_tgt.push_back(Eigen::Vector3f(x, y, z));
	}

	auto p = read_vector(vector_src);
	auto q = read_vector(vector_tgt);

	icp_warmup();

	auto start = std::chrono::steady_clock::now();
	auto [transformation, new_p] = icp(q, p, 100, 1e-6);
	auto end = std::chrono::steady_clock::now();	
	std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
	std::cout << duration.count() * 1000.0 << "ms" << std::endl;
	std::cout << transformation << std::endl;

	return 0;
}
