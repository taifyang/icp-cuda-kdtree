ICP algorithm implemented with cudaKDTree.

See [OPTIMIZATION.md](OPTIMIZATION.md) for the measured performance work, the CUDA/Eigen
pitfalls that shaped the current implementation, and how to reproduce the verification.
Note that the files in `data/` are binary PCD, which `main.cpp` cannot read.