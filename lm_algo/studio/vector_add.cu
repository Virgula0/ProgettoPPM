#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define STREAM_ARRAY_SIZE 1'000'000

/*
 * ============================================================================
 * GUDA RAPIDA AI QUALIFICATORI CUDA (__global__, __device__, __host__, __shared__)
 * ============================================================================
 *
 * 1. __global__ (Qualificatore di Funzione / Kernel)
 *    - Dove viene eseguita: GPU
 *    - Chi la chiama: CPU (tramite la sintassi <<<blocchi, thread>>>)
 *    - Tipo di ritorno: Deve essere SEMPRE 'void'
 *    - Scopo: È il punto di ingresso per il calcolo parallelo (Kernel).
 *    - Esempio:
 *        __global__ void mioKernel(float* d_data) { ... }
 *
 * 2. __device__ (Qualificatore di Funzione Ausiliaria)
 *    - Dove viene eseguita: GPU
 *    - Chi la chiama: Solo la GPU (da un kernel __global__ o altra __device__)
 *    - Scopo: Funzione di supporto/utility usata durante il calcolo su GPU.
 *    - Esempio:
 *        __device__ float calcolaQuadrato(float x) { return x * x; }
 *
 * 3. __host__ (Qualificatore di Funzione CPU)
 *    - Dove viene eseguita: CPU
 *    - Chi la chiama: CPU
 *    - Scopo: È il codice C++ standard. Se non si specifica alcun qualificatore,
 *             la funzione è implicitamente __host__.
 *    - Nota: Si può combinare "__host__ __device__" per generare due versioni
 *            della stessa funzione (una per CPU e una per GPU).
 *
 * 4. __shared__ (Qualificatore di Variabile / Memoria Condivisa)
 *    - Dove risiede: SRAM ad alta velocità interna allo Streaming Multiprocessor (SM)
 *    - Ambito/Visibilità: Tutti i thread appartenenti allo STESSO BLOCCO
 *    - Scopo: Cache veloce gestita dal programmatore per evitare accessi 
 *             ripetuti e lenti alla VRAM globale (Global Memory).
 *    - Esempio:
 *        __global__ void filtro() {
 *            __shared__ float cacheBlocco[256]; // Condivisa tra i thread del blocco
 *        }
 *
 * ============================================================================
 * TABELLA RIASSUNTIVA
 * ============================================================================
 * Qualificatore | Applicato a | Eseguito da | Chiamato da | Scopo
 * --------------+-------------+-------------+-------------+-------------------
 * __global__    | Funzione    | GPU         | CPU         | Kernel parallelo
 * __device__    | Funzione    | GPU         | GPU         | Utility per GPU
 * __host__      | Funzione    | CPU         | CPU         | Codice C++ normale
 * __shared__    | Variabile   | Memoria SM  | Thread Blocco| Cache veloce blocco
 * ============================================================================
 */

// Kernel CUDA: l'esecuzione sul singolo thread
__global__ void vectorAdd(const float* a, const float* b, float* c, float scalar, int n) {
    // Calcolo dell'ID globale del thread
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Controllo dei limiti dell'array (Guard Clause)
    if (gid >= n) return;

    __shared__ int test = 1; 
    // Corpo del calcolo eseguito in parallelo dal Warp
    c[gid] = a[gid] + scalar * b[gid];
}

int main() {
    std::cout << "Sizeof " << sizeof(float) << "\n";
    size_t bytes = STREAM_ARRAY_SIZE * sizeof(float);
    float scalar = 2.0f;

    // 1. Allocazione e Inizializzazione dati su HOST (CPU)
    std::vector<float> h_a(STREAM_ARRAY_SIZE, 1.0f); // Tutti 1.0
    std::vector<float> h_b(STREAM_ARRAY_SIZE, 3.0f); // Tutti 3.0
    std::vector<float> h_c(STREAM_ARRAY_SIZE, 0.0f); // Risultato

    // 2. Allocazione memoria su DEVICE (GPU VRAM)
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 3. Copia dati da RAM (Host) a VRAM (Device)
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // 4. Configurazione della Griglia CUDA
    int threadsPerBlock = 256; // 256 thread per blocco (multiplo di 32 per i Warp)
    int blocksPerGrid = (STREAM_ARRAY_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    // 5. Lancio del Kernel GPU
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, scalar, STREAM_ARRAY_SIZE);

    // Attesa del completamento del lavoro sulla GPU e controllo errori
    cudaDeviceSynchronize();

    // 6. Copia del risultato da VRAM (Device) a RAM (Host)
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // 7. Verifica dei risultati (Verifica su un paio di indici)
    std::cout << "Risultato h_c[0]: " << h_c[0] << " (Atteso: " << 1.0f + 2.0f * 3.0f << ")" << std::endl;
    std::cout << "Risultato h_c[999999]: " << h_c[999999] << " (Atteso: 7.0)" << std::endl;

    // 8. Pulizia della memoria VRAM
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
