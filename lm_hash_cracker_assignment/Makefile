build:
	nvcc lm_cracker_cuda.cu -o lm_hash_cuda -lnvidia-ml
	gcc -fopenmp lm_cracker_openmp.c -o lm_hash_openmp -lcrypto -O3
.PHONY: build