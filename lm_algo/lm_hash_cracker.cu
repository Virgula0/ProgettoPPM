#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MAX_PAD 14

// Visibile da tutti i kernel su GPU ad altissima velocità (cache L1)
// L'ho calcolata manualmente con altri script dentro la cartella studio
// poi l'ho embeddata staticamente cosi' non ho bisogno di ricalcolarla ad ogni esecuzione
// per le variabili globali non posso usare __shared__ perche' con __shared__ condivido 
// solo nei thread dello stesso blocco
__constant__ uint8_t NULL_HALF_BYTES[8] = {
    0xAA, 0xD3, 0xB4, 0x35, 0xB5, 0x14, 0x04, 0xEE
};

__constant__ unsigned char MAGIC_CONSTANT[8] = {'K', 'G', 'S', '!', '@', '#', '$', '%'};
__constant__ char FULL_CHARSET[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";

constexpr unsigned int HOST_CHARSET_SIZE = sizeof(FULL_CHARSET) - 1; // -1 per il null bytes
__constant__ unsigned int CHARSET_SIZE = HOST_CHARSET_SIZE;

#include "des/des_cuda_constants.cuh"
#include "des/des_gpu_functions.cu" 

__host__ void toUpper(char *str) {
    while (*str) {
        if (*str >= 'a' && *str <= 'z') {
            *str -= ('a' - 'A');
        }
        str++;
    }
}

__host__  bool checkHEXRange(char ch) {
    return (ch >= 0x30 && ch <= 0x39) 
        || (ch >= 0x41 && ch <= 0x46)
        || (ch >= 0x61 && ch <= 0x66);
}

__host__ bool checkValidHash(char *toCrack) {
    if (!toCrack || strlen(toCrack) != 32) {
        return false;
    }

    for (int i=0; i< 32; ++i){
        char ch = toCrack[i];

        if (!checkHEXRange(ch)) {
            return false;
        }
    }

    return true;
}

// device utility
__device__ void bytes_to_des_key(const uint8_t raw_7_bytes[7], uint8_t key_out[8]) {
    uint64_t b = 0;

    for (int i = 0; i < 7; i++) {
        b = (b << 8) | raw_7_bytes[i];
    }

    key_out[0] = (uint8_t)((b >> 49) & 0xFE); 
    key_out[1] = (uint8_t)((b >> 42) & 0xFE); 
    key_out[2] = (uint8_t)((b >> 35) & 0xFE); 
    key_out[3] = (uint8_t)((b >> 28) & 0xFE); 
    key_out[4] = (uint8_t)((b >> 21) & 0xFE); 
    key_out[5] = (uint8_t)((b >> 14) & 0xFE); 
    key_out[6] = (uint8_t)((b >> 7) & 0xFE);  
    key_out[7] = (uint8_t)((b << 1) & 0xFE);  
}

// Convert a hex string to raw bytes
__host__ void hex_to_bytes(const char *hex_str, uint8_t *bytes_out, size_t num_bytes) {
    for (size_t i = 0; i < num_bytes; i++) {
        sscanf(hex_str + 2 * i, "%02hhx", &bytes_out[i]);
    }
}

__device__ void des_encrypt_block(const uint8_t key_bytes[8], const uint8_t input_block[8], uint8_t output_block[8]) {
    // Converti la chiave da 8 byte in uint64 (big-endian), richiesto dalla libreria
    uint64 key = 0;
    for (int i = 0; i < 8; i++) {
        key = (key << 8) | key_bytes[i];
    }

    // Converti il blocco di input in uint64 (big-endian), richiesto dalla libreria
    uint64 input = 0;
    for (int i = 0; i < 8; i++) {
        input = (input << 8) | input_block[i];
    }

    // Esegui la cifratura DES
    uint64 output = encrypt_message_gpu(input, key);

    // Riconverti l'output in array di byte (big-endian)
    for (int i = 7; i >= 0; i--) {
        output_block[i] = output & 0xFF;
        output >>= 8;
    }
}

__device__ void hash_half_fast(const char *candidate, unsigned int candidate_len, uint8_t output_block[8]) {
    uint8_t padded[7] = {0}; // Inizializzato a zero (padding automatico)

    for (size_t i = 0; i < candidate_len && i < 7; i++) {
        char c = candidate[i];
        if (c >= 'a' && c <= 'z') c -= 32; // Uppercase
        padded[i] = (uint8_t)c;
    }

    uint8_t key_bytes[8];
    bytes_to_des_key(padded, key_bytes);
    des_encrypt_block(key_bytes, MAGIC_CONSTANT, output_block); // definita in OpenCL/inc_cipher_des.cl
}

/*
// Questa non esiste su GPU! Infatti andrebbe in heap stack overflow una cosa del genere
// Su GPU bisogna mappare il grande numero di thread sul lavoro da eseguire, invece su CPU questo rimpiazza la mancanza del grande
// numero di thread presenti su GPU. GPU deve usare un approccio Index-To-String ovvero basandoci sull'id del thread assegniamo il lavoro da 
// svolgere
__device__ int build_combo_dfs(char *current, int depth, int target_depth, const uint8_t target_bytes[8], char *found_match) {
    if (depth == target_depth) {
        current[depth] = '\0';
        uint8_t candidate_block[8];
        hash_half_fast(current, candidate_block);

        if (memcmp(candidate_block, target_bytes, 8) == 0) {
            strcpy(found_match, current);
            return 1; // Success
        }
        return 0;
    }

    size_t charset_len = strlen(FULL_CHARSET);
    for (size_t i = 0; i < charset_len; i++) {
        current[depth] = FULL_CHARSET[i];
        if (build_combo_dfs(current, depth + 1, target_depth, target_bytes, found_match)) {
            return 1;
        }
    }
    return 0;
}
*/

// FUNZIONE: indexToCandidate
// SCOPO: Mappare in modo univoco l'ID del thread a una combinazione di caratteri.
// 
// Invece di far generare a un singolo thread un albero di combinazioni tramite
// ricorsione (DFS, come faceva lm_optimized.c), ogni thread calcola indipendentemente la propria stringa 
// "candidata" partendo dal suo ID globale (index).
//
// Questo procedimento sfrutta la conversione di base (esattamente come convertire 
// un numero da decimale a binario o esadecimale). Qui la "base" è la lunghezza 
// del charset (es. 69 caratteri). L'algoritmo riempie la stringa da 
// destra verso sinistra (da i = len-1 fino a 0):
// 1. Trova il carattere corrente usando il resto della divisione (index % CHARSET_SIZE).
// 2. Aggiorna l'indice dividendo per la base (index / CHARSET_SIZE) per passare
//    alla posizione successiva.
// Questo garantisce che a indici diversi corrispondano sempre stringhe diverse,
// permettendo a migliaia di core della GPU di lavorare in parallelo senza 
// duplicare il lavoro o causare stack overflow da ricorsione.
__device__ void indexToCandidate(uint64_t index, int len, char *out_str) {
    // Da destra a sinistra
    for (int i = len - 1; i >= 0; i--) {
        out_str[i] = FULL_CHARSET[index % CHARSET_SIZE]; // Carattere in base alla posizione corrente del thread
        index /= CHARSET_SIZE;                            // Avanza alla posizione successiva
    }
    out_str[len] = '\0';
}

/*
 * ARCHITETTURA DI PARALLELIZZAZIONE E DISTRIBUZIONE DEL LAVORO SU GPU:
 *
 * La ricerca non viene eseguita tramite esplorazione ricorsiva (DFS), ma
 * dividendo lo spazio delle soluzioni per lunghezza della password (da 1 a 7).
 *
 * Per una data lunghezza L, il numero totale di combinazioni equivale a (N^L),
 * dove N rappresenta la dimensione del set di caratteri (es. CHARSET_SIZE = 69).
 *
 * Esempio di scala di lavoro:
 * - Lunghezza 1: 69^1  = 69 thread lanciati in parallelo.
 * - Lunghezza 2: 69^2  = 4.761 thread lanciati in parallelo.
 * - Lunghezza 7: 69^7  = ~7,5 trilioni di combinazioni.
 *
 * Assegnazione del carico:
 * Ogni thread GPU legge il proprio identificatore globale unico (gid) e invoca
 * la funzione `indexToCandidate`. Tramite conversione numerica in base 69, 
 * l'indice globale viene trasformato direttamente nella stringa candidata 
 * associata a quel singolo thread.
 *
 * Flusso operativo del singolo thread:
 * 1. Calcolo dell'indice assoluto della combinazione (start_index + gid).
 * 2. Mappatura dell'indice in stringa (es. Thread 42 -> Candidate "AE").
 * 3. Calcolo dell'hash DES per la stringa candidata generata.
 * 4. Confronto diretto con l'hash target per la verifica di corrispondenza.
 *
 * Questo approccio evita sovrapposizioni di memoria, elimina la ridondanza di 
 * calcolo tra core differenti e previene errori di stack overflow.

 __global__ void crack_half_kernel(
    const uint8_t target_bytes[8],  // [1] L'hash DES (8 byte) da decifrare
    char *cracked_out,              // [2] Buffer GPU dove scrivere il testo in chiaro trovato
    uint64_t start_index,           // [3] Offset iniziale della combinazione per questo lotto
    uint64_t total_work,            // [4] Numero totale di combinazioni per la lunghezza attuale
    int candidate_len,              // [5] Lunghezza della password attualmente sotto test (1-7)
    int *found_flag                 // [6] Flag intero condiviso per segnalare il successo
)
*/
__global__ void crackHalfKernel(
    const uint8_t target_bytes[8], 
    char *cracked_out, 
    uint64_t start_index, 
    uint64_t total_work,
    unsigned int candidate_len, 
    unsigned int *found_flag
) {
    // Calcolo dell'ID globale del thread
    uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;

    if (gid >= total_work || *found_flag != 0) return; //guard closure

    // check if the half is just a bunch of null bytes
    // Cast a uint64_t per verificare se tutti gli 8 byte sono 0x00 con un'unica istruzione
    // equivalente di memcmp di lm_optimized.c
    if (*(reinterpret_cast<const uint64_t*>(target_bytes)) == 0ULL) {
        if (atomicExch(found_flag, 1u) == 0) { // operazione atomicExch(found_flag, 1u) scrive 1 atomicamente nella VRAM, quindi scrive e controlla
            cracked_out[0] = '\0'; // Assegna dopo che atomicExch ha confermato che questo thread è il primo
        }
        return;
    }

    uint64_t combinationIndex = start_index + gid; // lo sliding window del thread
    
    // calcola il candidate 
    char candidate[8]; // non posso passargli candidate_len perche' nvcc si lamenta di allocazioni di stack dinamiche, per sicurezza alloco 8 perche' tanto non puo' essere piu' grande di 8
    indexToCandidate(combinationIndex, candidate_len, candidate);

    // encrypt candidate
    uint8_t candidate_block[8];
    hash_half_fast(candidate, candidate_len, candidate_block);

    // check against known target
    // (memcmp(candidate_block, target_bytes, 8) == 0) // non disponibile
    // Casto sia candidate_block che target_bytes a uint64_t* e dereferenzio
    if (*reinterpret_cast<const uint64_t*>(candidate_block) == *reinterpret_cast<const uint64_t*>(target_bytes)) {
         // scrive trovato se non e' gia' stato fatto da un altro thread, dubbio, impossibile che l'abbia trovato un altro thread, avrebbe indice diverso?
        if (atomicExch(found_flag, 1u) == 0) {
            // Copia il candidato trovato nel buffer di output
            for (unsigned int i = 0; i < candidate_len; i++) {
                cracked_out[i] = candidate[i];
            }
            cracked_out[candidate_len] = '\0'; // Terminatore di stringa C
        }
        return;
    }
    
    return;
}

__host__ void splitHash(const char *hash_hex, uint8_t target1[8], uint8_t target2[8]) {
    hex_to_bytes(hash_hex, target1, 8);
    hex_to_bytes(hash_hex + 16, target2, 8);
}

__host__ bool checkIfFound(unsigned int *foundFlag1, unsigned int *foundFlag2, unsigned int *h_flag1, unsigned int *h_flag2) {
    cudaMemcpy(h_flag1, foundFlag1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flag2, foundFlag2, sizeof(int), cudaMemcpyDeviceToHost);
    return (*h_flag1 == 1) && (*h_flag2 == 1);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("[ERR] No cracking hash provided\n");
        return -1;
    }

    char *toCrack = argv[1];

    if (!checkValidHash(toCrack)) {
        printf("[ERR] Provided hash seem a not valid LM HASH\n");
        return -1;
    }

    // make everything upper case
    toUpper(toCrack);
    printf("[!] Target hash loaded: %s\n", toCrack);

    // split hash in two helves and convert in bytes
    const unsigned int halveBytes = 8;
    uint8_t target1[halveBytes], target2[halveBytes]; 
    splitHash(toCrack, target1, target2);

    // Allocazione memoria su DEVICE (GPU VRAM)
    uint8_t *target1GPU = nullptr, *target2GPU = nullptr;
    char *crackedPassword1GPU = nullptr, *crackedPassword2GPU = nullptr;
    unsigned int *foundFlag1 = nullptr, *foundFlag2 = nullptr;
    
    cudaMalloc(&target1GPU, halveBytes);
    cudaMalloc(&target2GPU, halveBytes);
    cudaMalloc(&crackedPassword1GPU, halveBytes);
    cudaMalloc(&crackedPassword2GPU, halveBytes);
    cudaMalloc(&foundFlag1, sizeof(int));
    cudaMalloc(&foundFlag2, sizeof(int));
    
    // Necessario per int senno hanno garbage in memoria
    cudaMemset(foundFlag1, 0, sizeof(int));
    cudaMemset(foundFlag2, 0, sizeof(int));
    
    // Copia dati da RAM (Host) a VRAM (Device)
    cudaMemcpy(target1GPU, target1, halveBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(target2GPU, target2, halveBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 0;
    int minGridSize = 0;

    // Chiedi a CUDA la configurazione OTTIMALE per il tuo kernel sulla GPU in uso
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,       // Minimo numero di blocchi per saturare la GPU
        &threadsPerBlock,   // Numero di thread per blocco suggerito (es. 256 o 512)
        crackHalfKernel,      // Nome della funzione __global__ del kernel
        0,                  // Memoria dinamica shared (0 se non ne usi)
        0                   // Limite max di blocchi (0 = nessun limite)
    );

    printf("[Debug] ================== Configurazione dinamica CUDA ================== \n");
    printf("[Debug] - Thread per Blocco: %d\n", threadsPerBlock);

    cudaStream_t stream1, stream2; // 2 stream uno per ogni meta' di hash viene gestito in automatico dalla gpu, molto comodo
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    clock_t start = clock();
    
    unsigned int h_flag1 = 0;
    unsigned int h_flag2 = 0; 

    // inizio bruteforce len per len fino a 7, la GPU calcolera' propriamente i candidate ongi volta
    // Calcolo combinazioni totali per questa specifica lunghezza (69^len), le prime lunghezze sono banali perche' 68 e 4671 vengono sparate in un solo colpo
    // Eseguito ancora su CPU, fino a 7 perche' puo' essere massimo fino a 7 caratteri
    for (unsigned int len = 1; len <= 7; len++) { 
        uint64_t totalCombinations = pow(HOST_CHARSET_SIZE, len);

        // Calcolo dinamico di grid e block size in base a totalCombinations
        int blocksPerGrid = (totalCombinations + threadsPerBlock - 1) / threadsPerBlock;

        // Se la prima metà dell'hash non è ancora stata trovata, esegui il kernel
        if (h_flag1 == 0){
            // Lancio dei kernel riutilizzando gli STESSI stream
            crackHalfKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
                target1GPU, crackedPassword1GPU, 0, totalCombinations, len, foundFlag1
            );
        }

        // Se la seconda metà dell'hash non è ancora stata trovata, esegui il kernel
        if (h_flag2 == 0){
            crackHalfKernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(
                target2GPU, crackedPassword2GPU, 0, totalCombinations, len, foundFlag2
            );
        }

        // Sincronizzazione per controllare se la password è stata trovata in questa 'len'
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        // Verifica lo stato sulla CPU
        if (checkIfFound(foundFlag1, foundFlag2, &h_flag1, &h_flag2)) {
            break;
        }
    }

    // Pulizia stream
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    // Time taken
    clock_t end = clock();

    // Copia del risultato da VRAM (Device) a RAM (Host)
    char crackedFirstPart[8] = {0}, crackedSecondPart[8] = {0}; // 15 per il null byte delle stringhe in c
    cudaMemcpy(crackedFirstPart, crackedPassword1GPU, halveBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(crackedSecondPart, crackedPassword2GPU, halveBytes, cudaMemcpyDeviceToHost);

    char crackedPassword[15] = {0};

    // Concatenate results
    sprintf(crackedPassword, "%s%s", crackedFirstPart, crackedSecondPart);
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\n[!] Success! Cracked Password: %s\n", crackedPassword);
    printf("[!] Taken %.4f seconds\n", time_taken);

    cudaFree(target1GPU);
    cudaFree(target2GPU);
    cudaFree(crackedPassword1GPU);
    cudaFree(crackedPassword2GPU);
    cudaFree(foundFlag1);
    cudaFree(foundFlag2);
}