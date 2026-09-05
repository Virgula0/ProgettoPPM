build:
	g++ bloom_filter_openmp.cpp libraries/MurmurHash3.cpp -o bloom_filter_openmp -fopenmp -O3
.PHONY: build

run-perf:
	OMP_PROC_BIND="spread" OMP_PLACES=threads perf stat -e cache-references,cache-misses ./bloom_filter_openmp rockyou.txt parole_uniche.txt
.PHONY: run-perf

run:
	OMP_PROC_BIND="spread" OMP_PLACES=threads ./bloom_filter_openmp rockyou.txt parole_uniche.txt 
.PHONY: run

perf: build run-perf