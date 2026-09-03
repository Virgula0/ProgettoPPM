#include <string.h>
#include <time.h>

#include "des/des.cu"
#include "useful/gpu_stats.cu"

// su gpu
__constant__ unsigned char MAGIC_CONSTANT[8] = {'K', 'G', 'S', '!', '@', '#', '$', '%'};
__constant__ char FULL_CHARSET[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./"
                                   ":;<=>?@[\\]^_`{|}~ "; // analisi del caso peggiore => esecuzione
                                                          // dell'algoritmo con una password di 14 bytes
                                                          // contenente tutti spazi

constexpr unsigned int HOST_CHARSET_SIZE = sizeof(FULL_CHARSET) - 1; // -1 per il null byte
__constant__ unsigned int CHARSET_SIZE = HOST_CHARSET_SIZE;          // su gpu

__host__ void toUpper(char* str) {
    while (*str) {
        if (*str >= 'a' && *str <= 'z') {
            *str -= ('a' - 'A');
        }
        str++;
    }
}

__host__ bool checkHEXRange(char ch) {
    return (ch >= 0x30 && ch <= 0x39) || (ch >= 0x41 && ch <= 0x46) || (ch >= 0x61 && ch <= 0x66);
}

__host__ bool checkValidHash(char* toCrack) {
    if (!toCrack || strlen(toCrack) != 32) {
        return false;
    }

    for (int i = 0; i < 32; ++i) {
        char ch = toCrack[i];

        if (!checkHEXRange(ch)) {
            return false;
        }
    }

    return true;
}

// la funzione serve per convertire la chiave da usare per il des in un input
// compatibile des lavora con chiavi a 64 bits, ma solo 56 sono effettivamente
// usati per la chiave viene scartato un bit per ogni byte (usati solitamente
// come parita', nell'LM vengono invcece ignorati) il bit ignorato per ogni byte
// e' l'ultimo ottavo bit meno significativo (a destra)
__device__ void bytes_to_des_key(const uint8_t raw_7_bytes[7], uint8_t key_out[8]) {
    // preparo un nuovo indirizzo a 64 bit per contenere tutti i 64 bit in una
    // locazione sola ma b contiene solo 56 bit
    uint64_t b = 0;
#pragma unroll
    for (int i = 0; i < 7; i++) {
        b = (b << 8) | raw_7_bytes[i]; // shifto di 8 bit e concateno di volta in
                                       // volta, easy fin qui
    }

// per ogni byte droppo sempre quello in ultima posizione
// 0XFE = 11111110, l'ultimo bit e' 0
// il fatto e' che dobbiamo isolare 7 bit alla volta
#pragma unroll
    for (int i = 7; i >= 0; i--) {
        key_out[i] = (uint8_t)((b & 0x7F) << 1); // b & 0x7F prendo gli ultimi 7 bit, posi li sposto a
                                                 // sinistra di uno lasciando l'ultimo posto a 0
        b = b >> 7;                              // leggo i prossimi 7 bit
    }
}

// Convert a hex string to raw bytes
__host__ void hex_to_bytes(const char* hex_str, uint8_t* bytes_out, size_t num_bytes) {
    for (size_t i = 0; i < num_bytes; i++) {
        sscanf(hex_str + 2 * i, "%02hhx", &bytes_out[i]);
    }
}

__device__ void hash_half_fast(const char* candidate, unsigned int candidate_len, uint8_t output_block[8]) {
    uint8_t padded[7] = {0}; // Inizializzato a zero (padding automatico)

    for (size_t i = 0; i < candidate_len && i < 7; i++) {
        char c = candidate[i];
        if (c >= 'a' && c <= 'z')
            c -= 32; // Uppercase
        padded[i] = (uint8_t)c;
    }

    uint8_t key_bytes[8];
    bytes_to_des_key(padded, key_bytes);
    des_encrypt_block(key_bytes, MAGIC_CONSTANT, output_block);
}

/*
// Questa non esiste su GPU! Infatti andrebbe in heap stack overflow una cosa
del genere
// Su GPU bisogna mappare il grande numero di thread sul lavoro da eseguire,
invece su CPU questo rimpiazza la mancanza del grande
// numero di thread presenti su GPU. GPU deve usare un approccio Index-To-String
ovvero basandoci sull'id del thread assegniamo il lavoro da
// svolgere
__device__ int build_combo_dfs(char *current, int depth, int target_depth, const
uint8_t target_bytes[8], char *found_match) { if (depth == target_depth) {
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
        if (build_combo_dfs(current, depth + 1, target_depth, target_bytes,
found_match)) { return 1;
        }
    }
    return 0;
}
*/

// FUNZIONE: indexToCandidate
// SCOPO: Mappare in modo univoco l'ID del thread a una combinazione di
// caratteri.
//
// Invece di far generare a un singolo thread un albero di combinazioni tramite
// ricorsione (DFS, come faceva lm_optimized.c), ogni thread calcola
// indipendentemente la propria stringa "candidata" partendo dal suo ID globale
// (index).
//
// Questo procedimento sfrutta la conversione di base (esattamente come
// convertire un numero da decimale a binario o esadecimale). Qui la "base" è la
// lunghezza del charset (es. 69 caratteri). L'algoritmo riempie la stringa da
// destra verso sinistra (da i = len-1 fino a 0):
// 1. Trova il carattere corrente usando il resto della divisione (index %
// CHARSET_SIZE).
// 2. Aggiorna l'indice dividendo per la base (index / CHARSET_SIZE) per passare
//    alla posizione successiva.
// Questo garantisce che a indici diversi corrispondano sempre stringhe diverse,
// permettendo a migliaia di core della GPU di lavorare in parallelo senza
// duplicare il lavoro o causare stack overflow da ricorsione.
__device__ void indexToCandidate(uint64_t index, int len, char* out_str) {
    // Da destra a sinistra
    for (int i = len - 1; i >= 0; i--) {
        out_str[i] = FULL_CHARSET[index % CHARSET_SIZE]; // Carattere in base alla posizione
                                                         // corrente del thread
        index /= CHARSET_SIZE;                           // Avanza alla posizione successiva
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
 * dove N rappresenta la dimensione del set di caratteri (es. CHARSET_SIZE =
69).
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
    char *cracked_out,              // [2] Buffer GPU dove scrivere il testo in
chiaro trovato uint64_t start_index,           // [3] Offset iniziale della
combinazione per questo lotto uint64_t total_work,            // [4] Numero
totale di combinazioni per la lunghezza attuale int candidate_len, // [5]
Lunghezza della password attualmente sotto test (1-7) int *found_flag // [6]
Flag intero condiviso per segnalare il successo
)
*/
__global__ void crackHalfKernel(const uint8_t target_bytes[8], char* cracked_out, uint64_t start_index,
                                uint64_t total_work, unsigned int candidate_len, unsigned int* found_flag) {
    // Calcolo dell'ID globale del thread
    uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t idx = start_index + gid; // indice assoluto nel range [start_index, total_work-1]

    // senza idx con password da 7 byte si ha un bug infatti la dimensione della
    // griglia eccede il max soluzione: usare chunks

    if (idx >= total_work || *found_flag != 0)
        return; // guard closure

    // uint64_t combinationIndex = start_index + gid; // lo sliding window del
    // thread

    // calcola il candidate
    char candidate[8]; // non posso passargli candidate_len perche' nvcc si
                       // lamenta di allocazioni di stack dinamiche, per sicurezza
                       // alloco 8 perche' tanto non puo' essere piu' grande di 8
    indexToCandidate(idx, candidate_len, candidate);

    // encrypt candidate
    uint8_t candidate_block[8];
    hash_half_fast(candidate, candidate_len, candidate_block);

    // check against known target
    // (memcmp(candidate_block, target_bytes, 8) == 0) // non disponibile
    // Casto sia candidate_block che target_bytes a uint64_t* e dereferenzio
    if (*reinterpret_cast<const uint64_t*>(candidate_block) == *reinterpret_cast<const uint64_t*>(target_bytes)) {
        // scrive trovato se non e' gia' stato fatto da un altro thread, dubbio,
        // impossibile che l'abbia trovato un altro thread, avrebbe indice diverso?
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

__host__ void splitHash(const char* hash_hex, uint8_t target1[8], uint8_t target2[8]) {
    hex_to_bytes(hash_hex, target1, 8);
    hex_to_bytes(hash_hex + 16, target2, 8);
}

__host__ bool checkIfFound(unsigned int* foundFlag1, unsigned int* foundFlag2, unsigned int* h_flag1,
                           unsigned int* h_flag2) {
    cudaMemcpy(h_flag1, foundFlag1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flag2, foundFlag2, sizeof(int), cudaMemcpyDeviceToHost);
    return (*h_flag1 == 1) && (*h_flag2 == 1);
}

/*
__global__ void testDES(uint8_t* output) {
    // Simula la chiave per "TEST" (7 byte: T,E,S,T,0,0,0)
    uint8_t raw_key[7] = {'T', 'E', 'S', 'T', 0, 0, 0};
    uint8_t key[8];
    bytes_to_des_key(raw_key, key); // usa la funzione
    // Plaintext = MAGIC_CONSTANT
    uint8_t plain[8];
    for (int i = 0; i < 8; i++)
        plain[i] = MAGIC_CONSTANT[i];
    uint8_t cipher[8];
    des_encrypt_block(key, plain, cipher);
    // Copia il risultato in output
    for (int i = 0; i < 8; i++)
        output[i] = cipher[i];
}
*/

__host__ void cleanGPU(uint8_t* target1GPU, uint8_t* target2GPU, char* crackedPassword1GPU, char* crackedPassword2GPU,
                       unsigned int* foundFlag1, unsigned int* foundFlag2) {
    cudaFree(target1GPU);
    cudaFree(target2GPU);
    cudaFree(crackedPassword1GPU);
    cudaFree(crackedPassword2GPU);
    cudaFree(foundFlag1);
    cudaFree(foundFlag2);
}

int main(int argc, char** argv) {
    /*
      // debug algoritmo des per correttezza
      uint8_t *dev_output;
      cudaMalloc(&dev_output, 8);
      testDES<<<1,1>>>(dev_output);
      uint8_t host_output[8];
      cudaMemcpy(host_output, dev_output, 8, cudaMemcpyDeviceToHost);

      printf("[TEST] Cifrato di 'TEST' con MAGIC_CONSTANT: ");
      for (int i=0;i<8;i++) printf("%02X", host_output[i]);
      printf("\n");
      printf("ATTESO 01FC5A6BE7BC6929\n");
      return 1;
    */

    if (argc < 2) {
        printf("[ERR] No cracking hash provided\n");
        return -1;
    }

    char* toCrack = argv[1];

    if (!checkValidHash(toCrack)) {
        printf("[ERR] Provided hash seem a not valid LM HASH\n");
        return -1;
    }

    const uint8_t HOST_NULL_HALF[8] = {0xAA, 0xD3, 0xB4, 0x35, 0xB5, 0x14, 0x04, 0xEE};
    unsigned int h_flag1 = 0;
    unsigned int h_flag2 = 0;

    // make everything upper case
    toUpper(toCrack);
    printf("[!] Target hash loaded: %s\n", toCrack);

    // split hash in two helves and convert in bytes
    const unsigned int halfBytes = 8;
    uint8_t target1[halfBytes], target2[halfBytes];
    splitHash(toCrack, target1, target2);

    // Allocazione memoria su DEVICE (GPU VRAM)
    uint8_t *target1GPU = nullptr, *target2GPU = nullptr;
    char *crackedPassword1GPU = nullptr, *crackedPassword2GPU = nullptr;
    unsigned int *foundFlag1 = nullptr, *foundFlag2 = nullptr;

    cudaMalloc(&target1GPU, halfBytes);
    cudaMalloc(&target2GPU, halfBytes);
    cudaMalloc(&crackedPassword1GPU, halfBytes);
    cudaMalloc(&crackedPassword2GPU, halfBytes);
    cudaMalloc(&foundFlag1, sizeof(int));
    cudaMalloc(&foundFlag2, sizeof(int));

    // Necessario per int senno hanno garbage in memoria
    cudaMemset(foundFlag1, 0, sizeof(int));
    cudaMemset(foundFlag2, 0, sizeof(int));

    // Copia dati da RAM (Host) a VRAM (Device)
    cudaMemcpy(target1GPU, target1, halfBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(target2GPU, target2, halfBytes, cudaMemcpyHostToDevice);

    // check null bytes password
    if (memcmp(target1, HOST_NULL_HALF, 8) == 0) {
        printf("[OPTIMIZATION] NULL BYTES PASSWORD DETECTED\n");
        cleanGPU(target1GPU, target2GPU, crackedPassword1GPU, crackedPassword2GPU, foundFlag1, foundFlag2);
        return 0;
    }

    // Controllo se la seconda parte e' null bytes, evito di eseguire lo stream
    if (memcmp(target2, HOST_NULL_HALF, 8) == 0) {
        printf("[OPTIMIZATION] NULL HALF SECOND PART DETECTED, PASSWORD IS SHORTER "
               "THAN 8 CHARS...\n");
        h_flag2 = 1;
        unsigned int one = 1;
        cudaMemcpy(foundFlag2, &one, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemset(crackedPassword2GPU, 0, 1); // stringa vuota
    }

    int threadsPerBlock = 0;
    int minGridSize = 0;

    // Chiedi a CUDA la configurazione OTTIMALE per il kernel sulla GPU in uso
    cudaOccupancyMaxPotentialBlockSize(&minGridSize,     // Minimo numero di blocchi per saturare la GPU
                                       &threadsPerBlock, // Numero di thread per blocco suggerito (es. 256 o 512)
                                       crackHalfKernel,  // Nome della funzione __global__ del kernel
                                       0,                // Memoria dinamica shared (0 non usato)
                                       0                 // Limite max di blocchi (0 = nessun limite)
    );

    int maxGridDimX;
    cudaDeviceGetAttribute(&maxGridDimX, cudaDevAttrMaxGridDimX, 0);

    printf("[Debug] ================== Configurazione dinamica CUDA "
           "================== \n");
    printf("[Debug] - Thread per Blocco: %d\n", threadsPerBlock);
    printf("[Debug] - MaxGrid dimension: %d. Importante! questo causava un bug con "
           "le password lunghe 7 caratteri, perche' eccedevano il block size nelle "
           "combinazioni totali (69^7)\n",
           maxGridDimX);
    printf("[Debug] "
           "================================================================== \n");
    cudaStream_t stream1,
            stream2; // 2 stream uno per ogni meta' di hash viene gestito in
                     // automatico dalla gpu, molto comodo
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    clock_t startTime = clock();
    nvmlDevice_t nvmlDevice;
    bool nvmlReady = initNVML(&nvmlDevice);

    // inizio bruteforce len per len fino a 7, gli stream della GPU calcoleranno
    // propriamente i candidate di thread in thread Calcolo combinazioni totali
    // per questa specifica lunghezza (69^len), le prime lunghezze sono banali
    // perche' 68 e 4671 vengono sparate in un solo colpo Eseguito ancora su CPU,
    // fino a 7 perche' puo' essere massimo fino a 7 caratteri
    for (unsigned int len = 1; len <= 7; len++) {
        uint64_t totalCombinations = 1;

        for (unsigned int i = 0; i < len; i++) {
            totalCombinations *= HOST_CHARSET_SIZE;
        }

        printf("[DEBUG] Raggiunta lunghezza %u, combinazioni totali da testare %lu\n", len, totalCombinations);

        // Calcola la dimensione del chunk: quanti thread per lancio (blocchi *
        // threadsPerBlock) Utilizza il maxGridDimX per non superare il limite
        // hardware
        uint64_t maxBlocksPerLaunch = (uint64_t)maxGridDimX; // già ottenuto prima del for
        uint64_t chunkSize = maxBlocksPerLaunch * (uint64_t)threadsPerBlock;

        // Se il chunkSize supera totalCombinations, usa totalCombinations (per le
        // prime len e' molto utile)
        if (chunkSize > totalCombinations) {
            chunkSize = totalCombinations;
        }

        // Calcola il numero totale di chunk necessari (arrotondato per eccesso)
        uint64_t totalChunks = (totalCombinations + chunkSize - 1) / chunkSize;

        uint64_t start = 0;
        uint64_t chunkIndex = 0;

        // ciclo while per dividere il lavoro al di sotto delle dimensioni massime
        // del blocco supportato dalla gpu
        while (start < totalCombinations) {
            uint64_t remaining = totalCombinations - start;
            uint64_t workThisLaunch = (remaining < chunkSize) ? remaining : chunkSize;
            uint64_t blocks = (workThisLaunch + threadsPerBlock - 1) / threadsPerBlock;

            if (blocks == 0) {
                blocks = 1;
            }

            printf("[DEBUG] Lunghezza: %u, eseguendo blocco %lu di %lu (start=%lu, "
                   "work=%lu)\n",
                   len, chunkIndex + 1, totalChunks, start, workThisLaunch);

            // Se la prima metà dell'hash non è ancora stata trovata, esegui il kernel
            if (h_flag1 == 0) {
                // Lancio dei kernel riutilizzando gli STESSI stream
                crackHalfKernel<<<blocks, threadsPerBlock, 0, stream1>>>(target1GPU, crackedPassword1GPU, start,
                                                                         totalCombinations, len, foundFlag1);

                cudaError_t launchErr = cudaGetLastError();
                if (launchErr != cudaSuccess) {
                    printf("[ERR] Errore di lancio del kernel: %s\n", cudaGetErrorString(launchErr));
                }
            }

            // Se la seconda metà dell'hash non è ancora stata trovata, esegui il
            // kernel
            if (h_flag2 == 0) {
                crackHalfKernel<<<blocks, threadsPerBlock, 0, stream2>>>(target2GPU, crackedPassword2GPU, start,
                                                                         totalCombinations, len, foundFlag2);

                cudaError_t launchErr = cudaGetLastError();
                if (launchErr != cudaSuccess) {
                    printf("[ERR] Errore di lancio del kernel: %s\n", cudaGetErrorString(launchErr));
                }
            }

            // Sincronizzazione per controllare se la password è stata trovata in
            // questa 'len'
            cudaError_t syncErr = cudaStreamSynchronize(stream1);
            if (syncErr != cudaSuccess) {
                printf("[ERR] Errore durante l'esecuzione sulla GPU: %s\n", cudaGetErrorString(syncErr));
            }

            syncErr = cudaStreamSynchronize(stream2);
            if (syncErr != cudaSuccess) {
                printf("[ERR] Errore durante l'esecuzione sulla GPU: %s\n", cudaGetErrorString(syncErr));
            }

            if (nvmlReady) {
                printGPUStats(nvmlDevice);
            }

            // Verifica lo stato sulla CPU
            if (checkIfFound(foundFlag1, foundFlag2, &h_flag1, &h_flag2)) {
                break;
            }

            start += workThisLaunch;
            chunkIndex++;
        }

        if (h_flag1 && h_flag2) { // recheck
            break;
        }

        printf("[DEBUG] Crackata prima meta'? [%d]. Crackata seconda meta'? [%d] "
               "(0 = no, 1 = si)\n",
               h_flag1, h_flag2);
        printf("[DEBUG] Elapsed %.4f seconds\n", ((double)(clock() - startTime) / CLOCKS_PER_SEC));
    }

    if (nvmlReady) {
        nvmlShutdown();
    }

    // Pulizia stream
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    // Time taken
    clock_t endTime = clock();

    if (!h_flag1 && !h_flag2) {
        printf("ERROR: PASSWORD NOT FOUND\n");
        cleanGPU(target1GPU, target2GPU, crackedPassword1GPU, crackedPassword2GPU, foundFlag1, foundFlag2);
        return -1;
    }

    // Copia del risultato da VRAM (Device) a RAM (Host)
    char crackedFirstPart[9] = {0}, crackedSecondPart[9] = {0}; // 15 per il null byte delle stringhe in c
    cudaMemcpy(crackedFirstPart, crackedPassword1GPU, halfBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(crackedSecondPart, crackedPassword2GPU, halfBytes, cudaMemcpyDeviceToHost);
    crackedFirstPart[8] = '\0'; // terminatore stringa C
    crackedSecondPart[8] = '\0';
    char crackedPassword[17] = {0};

    // Concatenate results
    sprintf(crackedPassword, "%s%s", crackedFirstPart, crackedSecondPart);
    crackedPassword[16] = '\0';
    double time_taken = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    printf("\n[!] Success! Cracked Password: %s\n", crackedPassword);
    printf("[!] Taken %.4f seconds\n", time_taken);
    cleanGPU(target1GPU, target2GPU, crackedPassword1GPU, crackedPassword2GPU, foundFlag1, foundFlag2);
}