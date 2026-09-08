# ProgettoPPM
Parallel Programming Project: LM cracker (DES based old hash GPU pure bruteforcer) + Parallelized Bloom Filter with OpenMP and Asyncio

## LM Hash Cracker

LM is a very old hash algorithm based on DES (NIST fips46-3 version).

It is fairly well known to be particularly insecure nowadays, and its strange hashing procedure works like this:

1. The user’s password is converted to uppercase, this drastically reduces the total combination space-set and introduces bruteforce opportunities on modern hardware.
2. This password is null-padded to 14 bytes.
3. The “fixed-length” password is split into two 7-byte halves.
4. These values are used to create two DES keys, one from each 7-byte half, by converting the seven bytes into a bit stream and inserting a parity bit after every seven bits. This generates the 64 bits needed for the DES key.
5. Each of these keys is used to DES-encrypt the constant ASCII string `KGS!@#$%`, resulting in two 8-byte ciphertext values. Set the DES CipherMode to ECB and the PaddingMode to NONE.
6. These two ciphertext values are concatenated to form a 16-byte value, which is the LM hash.

You can find more info, try to produce new hashes and find a Java implementation at: https://asecuritysite.com/security50.aspx.

The following educational-purpose implementation of crackers has been implemented, highlighting time differences between:

- The CUDA implementation (as expected)
- The OpenMP (CPU) version.

Both follow C-style programming; however, CUDA is not pure C, even if it has been treated like that.

The most challenging part has surely been rewriting the DES implementation because most available CUDA implementations do not strictly follow the standard (NIST fips46-3: https://csrc.nist.gov/files/pubs/fips/46-3/final/docs/fips46-3.pdf) required by the `LM Hash` and trying to use `OpenCL` kernels by hashcat required more adaptation than expected so the decision was to proceed for a full re-implementation.

The charset proposed is composed of all printable characters and mostly matches those used as the full charset `?a` in hashcat.

```c
char FULL_CHARSET[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";
```

The worst case for both `OpenMP` and `CUDA` scenarios is a full `14-byte` provided hash of empty spaces (`0x20`), empirically calculated by brute-forcing it on `CUDA` and taking `~4.20 hours` against estimated (not empirically tested for obvious reasons) `~10.45 days` of `OpenMP` on the CPU version. It will follow, at the end of this section, a table which is particularly useful to underline differences between the two.

The hardware used to conduct these benchmarks was:

- CPU: `Intel i5 12600k (6 Cores - 12 Logical Threads)`
- GPU: `NVIDIA RTX 3060Ti - Ampere architecture, 4864 CUDA cores`
- RAM: `32Gb DDR4`

```bash
nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Fri_Apr_24_07:22:02_PM_PDT_2026
Cuda compilation tools, release 13.3, V13.3.33
Build cuda_13.3.r13.3/compiler.37862127_0

gcc --version
gcc (GCC) 16.1.1 20260430
Copyright (C) 2026 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

If `4.20` hours for a full 14 bytes character long password in the worse case seems a lot just think that the total combination to be computed are composed by the two helves splitted and summed up togheter so the total combination amounts to: (69^7  = ~7,5 trillions) * 2 = __~15 trillions of password in total__ considering the 69 charset length used. Of course this is an educational purpose, and on the same hardware, providing the hash to `Hashcat` cut down the computational time about of `~15 minutes` to test the all  __~15 trillions of password__ `~30 Giga Hashes per seconds`. This is due to advanced optimisation techniques provided by Hashcat kernels, such as `Bit Slicing`, etc., which are out of scope for this project.

| N | Plaintext | Hash | CUDA time | OPENMP time | GPU Throughput | CPU Throughput | Speedup | Time Saved (%) |
|---|-----------|------|-----------|-------------|----------------|----------------|---------|----------------|
| 1 | T | `417EAF50CFAC29C3AAD3B435B51404EE` | 0.0108 secondi | 0.0010 secondi | 6.39 KHash/s | 69.0 KHash/s | 0.0926 | 0.0% |
| 2 | TE | `CEC18980D4FFADA7AAD3B435B51404EE` | 0.0116 secondi | 0.0009 secondi | 0.410 MH/s | 5.29 MH/s | 0.0776 | 0.0% |
| 3 | TES | `F726D4121A092D9AAAD3B435B51404EE` | 0.0112 secondi | 0.0116 secondi | 29.33 MH/s | 28.32 MH/s | 1.0357 | 3.45% |
| 4 | TEST | `01FC5A6BE7BC6929AAD3B435B51404EE` | 0.0227 secondi | 0.4927 secondi | 0.999 GH/s | 46.01 MH/s | 21.70 | 95.39% |
| 5 | TEST1 | `E88D94D6EBD10FC7AAD3B435B51404EE` | 0.6309 secondi | 28.16 secondi | 2.479 GH/s | 55.54 MH/s | 44.64 | 97.76% |
| 6 | TEST12 | `50081C6A6EDD109BAAD3B435B51404EE` | 31.77 secondi | 32.07 minuti | 3.397 GH/s | 56.08 MH/s | 60.58 | 98.35% |
| 7 | TEST123 | `624AAC413795CDC1AAD3B435B51404EE` | 37.00 minuti | 1.536 giorni (stimato) | 3.354 GH/s | 56.08 MH/s | 59.81 | 98.33% |
| 8 | TEST123T | `624AAC413795CDC1417EAF50CFAC29C3` | 37.00 minuti | 1.536 giorni | ~3.35 GH/s | ~56.08 MH/s | ~59.81 | ~98.33% |
| 9 | TEST123TE | `624AAC413795CDC1CEC18980D4FFADA7` | 37.00 minuti | 1.536 giorni | ~3.35 GH/s | ~56.08 MH/s | ~59.81 | ~98.33% |
| 10 | TEST123TES | `624AAC413795CDC1F726D4121A092D9A` | 37.00 minuti | 1.536 giorni | ~3.35 GH/s | ~56.08 MH/s | ~59.81 | ~98.33% |
| 11 | TEST123TEST | `624AAC413795CDC101FC5A6BE7BC6929` | 37.00 minuti | 1.536 giorni | ~3.35 GH/s | ~56.08 MH/s | ~59.81 | ~98.33% |
| 12 | TEST123TEST1 | `624AAC413795CDC1E88D94D6EBD10FC7` | 37.01 minuti | 1.537 giorni | ~3.35 GH/s | ~56.08 MH/s | ~59.81 | ~98.33% |
| 13 | TEST123TEST12 | `624AAC413795CDC150081C6A6EDD109B` | 37.53 minuti | 1.559 giorni | ~3.35 GH/s | ~56.08 MH/s | ~59.82 | ~98.33% |
| 14 | TEST123TEST123 | `624AAC413795CDC1624AAC413795CDC1` | 1.233 ore | 3.073 giorni | ~3.35 GH/s | ~56.08 MH/s | ~59.81 | ~98.33% |

---

### Used Formula

### Number of candidates for length $L$ ($1 \le L \le 14$):

$$C(L) = 69^L, \quad \text{if } 1 \le L \le 7$$

$$C(L) = 69^7 + 69^{L-7}, \quad \text{if } 8 \le L \le 14$$

---

### Metrics and Performance

* **Throughput ($\text{hashes/second}$):**
  $$\text{Throughput} = \frac{C(L)}{T}$$
  *(where $T$ is the time in seconds)*

* **GPU Speedup relative to CPU:**
  $$S = \frac{T_{\text{CPU}}}{T_{\text{GPU}}}$$

* **Time Saved:**
  $$E = \max\left(0, \left(1 - \frac{1}{S}\right) \times 100\right)\% = \max\left(0, \left(1 - \frac{T_{\text{GPU}}}{T_{\text{CPU}}}\right) \times 100\right)\%$$

---

### Notes on estimated values for $L \ge 7$:

* **For $L = 7$:** 
  $$T_{\text{CPU}} = T_{\text{CPU}}(L=6) \times \left(\frac{69^7}{69^6}\right) = 1924.38 \times 69 = 132782.22 \text{ s} \approx 1.536 \text{ days}$$
* **For $L > 7$:** 
  $$T_{\text{CPU}} = T_{\text{CPU}}(\text{first half at } 7 \text{ characters}) + T_{\text{CPU}}(\text{second half of length } L-7)$$
* Throughputs for $L \ge 5$ are practically constant because time scales linearly with $C(L)$.

## Parallelized Bloom Filter

Bloom Filters are space-efficient data strutcures that are primarily used as a mean to check if an element is part of a set. Since it is space-efficient, it is usually used with a great volume of data (e.g during a sign in it is used to check if the given password is acceptable or not). The objective is to understand how much parallelization will help the usage of a Bloom Filter, to do this, two versions Bloom Filter will be evaluated, one coded using C++ and OpenMP for parallelization, and the other coded with Python free-threaded version (experimental verion that permit to deactivate the GIL and achive real multithreading).

### How a Bloom Filter works

Generally, a Bloom Filter is an array of bits (could also be a boolean or byte array) initialized to 0. To insert an element, the element first needs to be converted into indices through the use of k hash functions, returning k indices. Once converted, the positions in the bit array indicated by the indices are set to 1 (this will lose all information about the element). The check works in the same way: the element to be searched is converted into indices, and then the Bloom Filter checks whether the value stored in those positions is 1 or 0. If all positions contain 1s, the element is part of the set; otherwise, if even one 0 is present, the element is not part of the set.

#### Fake Positives

In a Bloom Filter there cannot be false negatives but there can be false positives, the probability of a false positive is given by:

$$p = \left(1 - \left[1 - \frac{1}{m}\right]^{kn}\right)^{k}$$ or approximated $$p = \left(1 - e^{\frac{-kn}{m}}\right)^{k}$$

This probability needs to be reduced to a minimum to have a good Bloom Filter.

### Data used and performance evaluation variables

The rockyou.txt file has been used as dictionary and parole_uniche.txt, that is formed by words not present inside rockyou.txt composed by 8 randomly chosen letters; parole_uniche.txt has been used in the "contains" operation to test if the false positives follow the theoretical probability.
A Byte array has been utilized for the Bloom Filter using a bit packing strategy to try and mitigate bottlenecks caused by memory accesses.

* rockyou.txt - 14344377 passwords - ~140MB
* parole_uniche.txt - 999997 passwords - ~9MB

The performance attributes measured for both implementations are:

* Execution Time
* Speedup = $$\frac{executionTimeSequential}{executionTimeParallel}$$
* Efficiency = $$\frac{speedup}{numberOfThreads}$$

#### C++ OpenMP results

The only parallelizable parts of a Bloom Filter are the operations "add" and "contains", and so we parallelized them using pragma directives. Once obtained the parallelized version of the Bloom Filter we compare the results with the results of a sequential implementation of Bloom Filter, the results, of 15 test iterations, are:

| Ciclo | Tempo Seq. ADD (s) | Tempo Par. ADD (s) | Speedup ADD | Efficiency ADD | Tempo Seq. CONTAINS (s) | Tempo Par. CONTAINS (s) | Speedup CONTAINS | Efficiency CONTAINS | Verifica (Seq/Par) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.75601 | 0.764264 | 2.29765x | 28.7206% | 0.105339 | 0.0245089 | 4.29799x | 53.7249% | 8433 / 8433 |
| 2 | 1.76394 | 0.757237 | 2.32945x | 29.1181% | 0.116195 | 0.0245146 | 4.73984x | 59.2479% | 8433 / 8433 |
| 3 | 1.71722 | 0.746605 | 2.30004x | 28.7505% | 0.105972 | 0.0226688 | 4.6748x | 58.435% | 8433 / 8433 |
| 4 | 1.82079 | 0.76255 | 2.38776x | 29.8471% | 0.105332 | 0.0278888 | 3.77687x | 47.2109% | 8433 / 8433 |
| 5 | 1.74755 | 0.732118 | 2.38698x | 29.8372% | 0.105239 | 0.0227223 | 4.63153x | 57.8942% | 8433 / 8433 |
| 6 | 1.7584 | 0.732195 | 2.40155x | 30.0194% | 0.105167 | 0.0226926 | 4.63442x | 57.9302% | 8433 / 8433 |
| 7 | 1.75991 | 0.732689 | 2.40199x | 30.0249% | 0.105886 | 0.0238899 | 4.43227x | 55.4034% | 8433 / 8433 |
| 8 | 1.7578 | 0.732345 | 2.40024x | 30.0029% | 0.105518 | 0.0226658 | 4.6554x | 58.1925% | 8433 / 8433 |

| Operazione | Tempo Medio Sequenziale (s) | Tempo Medio Parallelo (s) | Speedup Medio | Efficiency Media |
|---|---|---|---|---|
| ADD | 1.76491 | 0.741147 | 2.38184x | 29.7731% |
| CONTAINS | 0.106123 | 0.0238917 | 4.47243x | 55.9054% |

A notable speedup is reached parallelizing "contains" with an acceptable level of efficiency, while for "add" the speedup and efficiency are worse. This is caused by the bit packing technique that requires and atomic operation for writing in the byte array.

#### Python free-threaded results

Same as the C++ OpenMP implementation, the parallelizable parts are the operations "add" and "contains". The parallelization has been obtained using Python native threads with the standard library module 'multiprocessing.pool.ThreadPool'.

| Ciclo | Tempo Seq. ADD (s) | Tempo Par. ADD (s) | Speedup ADD | Efficiency ADD | Tempo Seq. CONTAINS (s) | Tempo Par. CONTAINS (s) | Speedup CONTAINS | Efficiency CONTAINS | Verifica (Seq/Par) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 84.850860 | 90.787828 | 0.93x | 11.68% | 2.698682 | 1.393695 | 1.94x | 24.20% | 8539 / 8539 |
| 2 | 84.873115 | 92.670817 | 0.92x | 11.45% | 2.676591 | 1.157674 | 2.31x | 28.90% | 8539 / 8539 |
| 3 | 85.201686 | 75.509156 | 1.13x | 14.10% | 2.657308 | 1.071891 | 2.48x | 30.99% | 8539 / 8539 |
| 4 | 84.833558 | 72.810213 | 1.17x | 14.56% | 2.662760 | 1.038046 | 2.57x | 32.06% | 8539 / 8539 |
| 5 | 84.503808 | 72.435889 | 1.17x | 14.58% | 2.667830 | 1.078548 | 2.47x | 30.92% | 8539 / 8539 |
| 6 | 85.490919 | 91.068162 | 0.94x | 11.73% | 2.689181 | 1.186147 | 2.27x | 28.34% | 8539 / 8539 |
| 7 | 84.756644 | 98.812774 | 0.86x | 10.72% | 2.660590 | 1.048506 | 2.54x | 31.72% | 8539 / 8539 |
| 8 | 85.274003 | 90.334019 | 0.94x | 11.80% | 2.680425 | 1.074056 | 2.50x | 31.20% | 8539 / 8539 |
| 9 | 85.031182 | 85.598837 | 0.99x | 12.42% | 2.650066 | 1.061549 | 2.50x | 31.21% | 8539 / 8539 |
| 10 | 85.119402 | 73.845148 | 1.15x | 14.41% | 2.668620 | 1.124635 | 2.37x | 29.66% | 8539 / 8539 |
| 11 | 84.829701 | 71.654528 | 1.18x | 14.80% | 2.661025 | 0.986704 | 2.70x | 33.71% | 8539 / 8539 |
| 12 | 84.978749 | 72.504815 | 1.17x | 14.65% | 2.657608 | 1.044585 | 2.54x | 31.80% | 8539 / 8539 |
| 13 | 85.251395 | 90.167100 | 0.95x | 11.82% | 2.678232 | 1.150026 | 2.33x | 29.11% | 8539 / 8539 |
| 14 | 85.026555 | 88.767542 | 0.96x | 11.97% | 2.674827 | 1.132737 | 2.36x | 29.52% | 8539 / 8539 |
| 15 | 84.867235 | 90.416265 | 0.94x | 11.73% | 2.660707 | 0.960535 | 2.77x | 34.63% | 8539 / 8539 |

| Operazione | Tempo Medio Sequenziale (s) | Tempo Medio Parallelo (s) | Speedup Medio | Efficiency Media |
|---|---|---|---|---|
| ADD | 84.992587 | 83.825540 | 1.03x | 12.83% |
| CONTAINS | 2.669630 | 1.100622 | 2.44x | 30.53% |

HHere the only speedup has been obtained in the "contains" operation, with a difference of 1.56s between the sequential and parallel execution.
The "add" operation, on the other hand, practically equal in execution time with the parallel implementation, having a 1x speedup with very poor efficiency. This is caused not only by the use of locks to write in the byte array, but also by the use of locks in the dynamic assignment of data chunks to be processed and by the intrinsic overhead caused by Python itself.