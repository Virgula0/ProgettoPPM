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