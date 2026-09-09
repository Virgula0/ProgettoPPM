build:
	nvcc lm_cracker_cuda.cu -o lm_hash_cuda -lnvidia-ml
	gcc -fopenmp lm_cracker_openmp.c -o lm_hash_openmp -lcrypto -O3
.PHONY: build

# run with make perf hash="HASH"
# requires sudo pacman -S nsight-compute
perf:
	ncu --set full ./lm_hash_cuda $(hash)
.PHONY: build perf