#include <cstdint>
#include <cuda_runtime.h>
#include <stdint.h>

__constant__ const uint8_t IP_TABLE[64] = {58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4,
                                           62, 54, 46, 38, 30, 22, 14, 6, 64, 56, 48, 40, 32, 24, 16, 8,
                                           57, 49, 41, 33, 25, 17, 9,  1, 59, 51, 43, 35, 27, 19, 11, 3,
                                           61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7};

// This is to apply IP^-1, pass this to permute function and it will perform the
// inverse operation
__constant__ const uint8_t FP_TABLE[64] = {40, 8, 48, 16, 56, 24, 64, 32, 39, 7, 47, 15, 55, 23, 63, 31,
                                           38, 6, 46, 14, 54, 22, 62, 30, 37, 5, 45, 13, 53, 21, 61, 29,
                                           36, 4, 44, 12, 52, 20, 60, 28, 35, 3, 43, 11, 51, 19, 59, 27,
                                           34, 2, 42, 10, 50, 18, 58, 26, 33, 1, 41, 9,  49, 17, 57, 25};

__constant__ const uint8_t PC1_TABLE[56] = {57, 49, 41, 33, 25, 17, 9,  1,  58, 50, 42, 34, 26, 18, 10, 2,  59, 51, 43,
                                            35, 27, 19, 11, 3,  60, 52, 44, 36, 63, 55, 47, 39, 31, 23, 15, 7,  62, 54,
                                            46, 38, 30, 22, 14, 6,  61, 53, 45, 37, 29, 21, 13, 5,  28, 20, 12, 4};

__constant__ const uint8_t PC2_TABLE[48] = {14, 17, 11, 24, 1,  5,  3,  28, 15, 6,  21, 10, 23, 19, 12, 4,
                                            26, 8,  16, 7,  27, 20, 13, 2,  41, 52, 31, 37, 47, 55, 30, 40,
                                            51, 45, 33, 48, 44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32};

__constant__ const uint8_t SHIFTS[16] = {1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1};

__constant__ const uint8_t E_TABLE[48] = {32, 1,  2,  3,  4,  5,  4,  5,  6,  7,  8,  9,  8,  9,  10, 11,
                                          12, 13, 12, 13, 14, 15, 16, 17, 16, 17, 18, 19, 20, 21, 20, 21,
                                          22, 23, 24, 25, 24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32, 1};

__constant__ const uint8_t P_TABLE[32] = {16, 7, 20, 21, 29, 12, 28, 17, 1,  15, 23, 26, 5,  18, 31, 10,
                                          2,  8, 24, 14, 32, 27, 3,  9,  19, 13, 30, 6,  22, 11, 4,  25};

// Composed by S1, S2, implements the selection function
// per qualche motivo se metto __constant__ le S_BOXES ci sono netti cali di prestazioni da parte della GPU
__device__ const uint8_t S_BOXES[8][4][16] = {
        // S1
        {{14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7},
         {0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8},
         {4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0},
         {15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13}},
        // S2
        {{15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10},
         {3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5},
         {0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15},
         {13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9}},
        // S3
        {{10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8},
         {13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1},
         {13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7},
         {1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12}},
        // S4
        {{7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15},
         {13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9},
         {10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4},
         {3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14}},
        // S5
        {{2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9},
         {14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6},
         {4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14},
         {11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3}},
        // S6
        {{12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11},
         {10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8},
         {9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6},
         {4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13}},
        // S7
        {{4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1},
         {13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6},
         {1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2},
         {6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12}},
        // S8
        {{13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7},
         {1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2},
         {7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8},
         {2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11}}};

// Funzione ausiliaria per la permutazione dinamica dei bit
// in (uint64_t): Il blocco di dati da permutare. Deve essere un singolo intero
// a 64 bit, non un array di puntatori a 8 bit (uint8_t[8]) out_bits (int): Il
// numero di bit attesi in output. Per l'Initial Permutation (IP) del DES è 64.
// in_bits (int): Il numero di bit in input. Anche questo per l'IP è 64.
__device__ __forceinline__ uint64_t permute(uint64_t input, const uint8_t* table, int out_bits, int in_bits) {
    uint64_t out = 0;

#pragma unroll
    for (int i = 0; i < out_bits; i++) {
        uint8_t pos_src = table[i] - 1;
        unsigned int shift_in = (in_bits - 1) - pos_src;
        uint64_t bit = (input >> shift_in) & 1ULL; // 1uLL un uno a 64 bit 000000 .... 00001
        unsigned int shift_out = (out_bits - 1) - i;
        out = (bit << shift_out) | out; // oppure |= (bit << shift_out)
    }
    return out;
}

// generate_subkeys rappresenta la Key schedule calculation
__device__ __forceinline__ uint64_t generate_subkey(int shift_index, uint32_t *D, uint32_t *C) {
    // il numero di shift avviene in base al numero di feistel round (ovvero 16)
    // adesso applichiamo una circular left operation su C e su D
    // For Feistel round 1, 2, 9, and 16 both halves (left and right) undergo 1-bit left shift operation. 
    // For others rounds (3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15) the halves undergo 2-bit left shift operation. 
    // La tabell shifts e' la lookup table per eseguire il numero di shift necessario
    uint8_t shift_to_apply = SHIFTS[shift_index];
    // ad entrambe le meta'
    // circular shift
    // Vuol dire che shiftato in bit (o 2 bit a sinistra vengono) vengono rimessi sul fondo
    // uint32_t get_first_elements_mask = shift_to_apply == 1 ? 0x8000000 : 0xC000000; // se 1 e' lo shift allora maschera sara' 1000000000000000000000000000 altrimenti 1100000000000000000000000000
    // uint32_t drop_first_elements_mask = shift_to_apply == 1 ? 0x7FFFFFF : 0x3FFFFFF; // se 1 e' applichiamo la mashera 0111111111111111111111111111 altrimenti 0011111111111111111111111111
    // *D = ((*D & drop_first_elements_mask) << shift_to_apply) | ((*D & get_first_elements_mask) >> (28 - shift_to_apply)); // prima applico lo shift a sinistra ma poi devo reinserire quei bit in fondo 
    // *C = ((*C & drop_first_elements_mask) << shift_to_apply) | ((*C & get_first_elements_mask) >> (28 - shift_to_apply));
    // il ragionamento delle righe precedenti era corretto ma posso semplificare, ci sono delle branch decision: bad per la gpu 
    // infatti posso prima shiftare di shift_to_apply e poi applicare la maschera 1111111111111111111111111111 per prendere i 28 bits tutto alla fine
    // in questo modo non ho bisogno dei condizionali get_first_elements_mask e drop_first_elements_mask che introducono branch decision sulla gpu
    *D = ((*D << shift_to_apply) | (*D >> (28 - shift_to_apply))) & 0xFFFFFFF;
    *C = ((*C << shift_to_apply) | (*C >> (28 - shift_to_apply))) & 0xFFFFFFF; 
    // dopo l'applicazione dello shift C e D sono ricombinati in un blocco unico da 56 bit
    // remerge
    // uint64_t shifted_remerged = ((uint64_t)(*C) << 28) | *D; OTTIMIZZAZIONE: non alloco una variabile appositamente per questo, lo passo direttamente alla function
    // adesso il blocco ricombinato da 56 bit viene passato ad una permutation choice PC2_TABLE
    return permute((((uint64_t)(*C) << 28) | *D), PC2_TABLE, 48, 56); // ritorna i 48 bits necessari
}

// Rappresenta la funzione f, prende in input 32 bits (da R) e 48-bits dalla
// chiave OTTIMIZZAZIONE: __forceinline__ per il compilatore
__device__ __forceinline__ uint32_t mangler_cipher_function(uint32_t r_block, uint64_t bit_48_key) {
    // plaintext expansion (E_TABLE) 32 bits expanded to 48 bits
    uint64_t expanded_plaintext = permute(r_block, E_TABLE, 48, 32);

    // xored with 48 bits of the subkey
    uint64_t xored_plaintext = expanded_plaintext ^ bit_48_key; // still 48 bits

    // split into eight chunks of 6-bit size each
    uint8_t chunks[8] = {0};

    // uint64_t temp_xored = xored_plaintext; // se xored_plaintext non e' riusato
                                           // possiamo eliminarlo successivamente, OTTIMIZZAZIONE: uso direttamente xored_plaintext
#pragma unroll
    for (int i = 7; i >= 0; i--) {                // big-endian itero al contrario
        chunks[i] = (uint8_t)(xored_plaintext & 0x3F); // prendo gli ultimi 6 bit meno
                                                  // significativi ad eccezione dei primi
                                                  // due (perche' sto operando su uint8_t)
        // spiegazione della maschera 0x3F , in binario e' 0011 1111 ovvero gli
        // ultimi 6 bit
        xored_plaintext = xored_plaintext >> 6;
    }

    // substitution (S_BOXES)
    // ogni chunk lo do ad una S_BOX differente S1 ad S8
    // riducono ognuno dei chunk da 6 bit in 4 bit
    // il trucco sta nell'utilizzare le S-Box come lookup tables, ulitizzando i
    // bit di ogni chunk per interpretare le coordinate nella matrice
    // tridimensionale S_BOXES[8][4][16]. Ogni chunk e' formato da [b1 b2 b3 b4 b5
    // b6]  S_BOXES[INDICE_SBOX][RIGA][COLONNA] Riga (2 bit): È formata dal 1° e
    // dal 6° bit (il primo e l'ultimo). Colonna (4 bit): È formata dai 4 bit
    // centrali (b2 b3 b4 b5).
    uint32_t combined_chunks = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) { // ogni S-Box da applicare ad ogni chunk
        // suppongo che chunks[i] sia 101010
        // 101010 chunks[i] & 0x20 (100000 = 32 in decimale ovvero 0x20 in hex) per
        // prendere il primo 1 ovvero ottengo 100000 (occhio va shiftato di 4
        // perche' cosi' corrisponde a 64 in decimale) 101010 chunks[i] & 0x1
        // (000001 = 1 in decimale ovvero 0x1 in hex) per prendere il primo l'ultimo
        // una volta estratti ottengo 10 in ordine, per concatenarli uso << con |,
        // il tutto diventa
        uint8_t row =
                ((chunks[i] & 0x20) >> 4) | (chunks[i] & 0x1); // lo shift serve per spostarlo a destra di 4 posti dato
                                                               // che si trova in posizione e poi lo concateno con l'or
                                                               // all'altro, avendolo shiftato solo di 4 ottengo 000010
                                                               // con un posto a destra per concatenare l'altro
        // suppongo che chunks[i] sia 101010
        // 101010 chunks[i] & 0x10 (010000)
        // 101010 chunks[i] & 0x8 (001000)
        // 101010 chunks[i] & 0x4 (000100)
        // 101010 chunks[i] & 0x2 (000010)
        // vanno shiftati a destra tutti di 1 perche' l'ultimo bit lo elimino
        // uint8_t column = ((chunks[i] & 0x10) >> 1) | ((chunks[i] & 0x8) >> 1) |
        // ((chunks[i] & 0x4) >> 1) | ((chunks[i] & 0x2) >> 1); questa operazione e'
        // corretta ma si puo' fare direttamente:
        uint8_t column = (chunks[i] >> 1) & 0x0F; // 0x0F e' (0000 1111) cosi' shiftando elimino quello
                                                  // meno significativo e con la maschera elimino anche
                                                  // il primo bit ottenendo i 4 che mi servono
        uint8_t val = S_BOXES[i][row][column];    // valore a 4 bit ottenuto di ritorno
        // adesso combiniamo di nuovo tutti i chunks per ottenere 32 bit finali
        combined_chunks = (combined_chunks << 4) | val; // sposto di 4 i precedenti poi mergio con i successivi
    }

    // final permutation (P_TABLE) also called transposition
    return (uint32_t)permute(combined_chunks, P_TABLE, 32,
                             32); // convertendo in uint32_t taglio i 32 bit piu'
                                  // significativi che sono tutti 0 tanto
}

// keys_bytes => bytes of the key
// input_block => plaintext to encrypt
// outout_block => encrypted block
// Algoritmo preso dalle spcifiche NIST (fips46-3) versione usata dall'LM HASH:
// https://csrc.nist.gov/files/pubs/fips/46-3/final/docs/fips46-3.pdf
// https://www.geeksforgeeks.org/computer-networks/data-encryption-standard-des-set-1/
// per aiutarsi con una spiegazione piu' semplice
__device__ __forceinline__ void des_encrypt_block(const uint8_t key_bytes[8], const uint8_t input_block[8],
                                                  uint8_t output_block[8]) {
    // converto il blocco in input in un numero rappresentativo a 64 bit
    // notazione a nell'ordine BigEndian
    uint64_t block64 = 0;

#pragma unroll
    for (int i = 0; i < 8; i++) {
        // quando faccio operazioni tra block64 e input_block, ad esempio se il
        // plaintext e' AAAAA.. ovvero 8 bytes ogni byte e' 8 bit, il compilatore in
        // automatico estende gli 8 bit a 64 bit quindi input_block[0] = 00000000
        // ... 01000001 (supponendo che sia una A = 65 in decimale = 01000001 in
        // binario) l'operazione OR avviene tra due blocchi di 64 bits ogni volta
        // che shifta viene spostato di 8 bits quindi se all'inizio e' 00000000 ...
        // 000000001000001 => 00000000 ... 100000100000000 e cosi' arrivando a
        // popolare l'intero da 64 bit con tutti i bit 1000001 della lettera in
        // questione (o delle lettere se sono diverse in questo caso l'esempio era
        // con 8 byte di A) con i bit della lettere del plaintext in unico indirizzo
        // molto efficiente
        block64 = (block64 << 8) | input_block[i];
    }

    // IP = initial permutation
    // Initial permutation
    uint64_t permuted_block = permute(block64, IP_TABLE, 64, 64);

    // genera subkeys
    // uint64_t subkeys[16];
    // converti la chiave in int64 bits
    uint64_t key64 = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        key64 = (key64 << 8) | key_bytes[i];
    }

    // Prima operazione: permutazione con tabella PC-1 (su geek for geeks non corrisponde con quella del NIST), produce da 64 a 56 bits
    // Ottimizzazione applicata da me per la funzione generate_keys, invece di eseguire un altro ciclo con la generazione delle 16 chiavi eseguo tutto 
    // nello stesso feistel cicle, quindi qui parte la generazione delle subkeys, l'inizializzazione di C e D basta sia eseguita la prima volta
    uint64_t bit_56_key = permute(key64, PC1_TABLE, 56, 64);

    // Split della chiave in 2 sottoinsiemi da 18 bits ciascuna chiamate C(primi 28 bits) e D(ultimi 28 bits)
    uint32_t D = bit_56_key & 0xFFFFFFF; // applico maschera e prendo gli ultimi 28 bits 1111111111111111111111111111
    uint32_t C = (bit_56_key >> 28 ) & 0xFFFFFFF; // shifto di 28 e prendo gli ultimi 28 bits2

    /*
        // Splitting
        // uint32_t L[17] = {0}; // contiene da L[0] .. L[16]
        // uint32_t R[17] = {0}; // contiene da R[0] .. R[16]
        // Metà SINISTRA (32 bit più significativi)
        // L[0] = (uint32_t)(permuted_block >> 32); // qui va bene lo shift
        // Metà DESTRA (32 bit meno significativi)
        // R[0] = (uint32_t)permuted_block; // se faccio << 32 ottengo tutti 0
       perche, conversione a uint32_t taglia automaticamente la meta' di sinistra
       prende usa little-endian come precedenza
        // L e R sono blocchi da 32 bits
        // Complex permutation key dependant (composed by a enchiphering function
       and a function KS called key schedule, cioe' quella che genera chiavi)
        // Input block is composed by LR = 64 bits
        // K is a block of 48bits chosen from 64 bits input key
        // L'R' is the output obtained from input L and R generic is
        // L' = R
        // R' = L XOR f(R,K) (where OP is bit-by-bit addition % 2 = XOR), 1 XOR 0
       = 1


        // Version funzionante ma piu' lenta per accesso agli indici dinamici
       degli array for (int i=1; i <= 16; i++) { // feistel cicle, devono essere
       16 round esatti non 15! bug! uint64_t currentSubkey = subkeys[i-1]; // sono
       gia' a 48 bits uint32_t f_function_result = mangler_cipher_function(R[i-1],
       currentSubkey); // ritorna 32 bit
            // swap L and R
            R[i] = L[i-1] ^ f_function_result;
            L[i] = R[i-1];
        }
    */

    // Metà SINISTRA (32 bit più significativi)
    uint32_t L_prev = (uint32_t)(permuted_block >> 32); // qui va bene lo shift
    // Metà DESTRA (32 bit meno significativi)
    uint32_t R_prev = (uint32_t)permuted_block; // se faccio << 32 ottengo tutti 0 perche, conversione a
                                                // uint32_t taglia automaticamente la meta' di sinistra
                                                // prende usa little-endian come precedenza

#pragma unroll
    for (int i = 0; i < 16; i++) {
        uint64_t current_subkey = generate_subkey(i, &D, &C);
        uint32_t temp = R_prev; // temp serve per lo swap tra L e R
        R_prev = L_prev ^ mangler_cipher_function(R_prev, current_subkey);
        L_prev = temp;
    }

    /*
    alla fine dell'algoritmo, avviene un pre-output
    After these 16 rounds we get two blocks (Left and Right) of 32-bit each.
    The two 32-bit halves are again swapped back, resulting in a 64-bit block.
    This step is called 32-bit Swap in DES encryption algorithm.
    // L[16] contiene R[15] e R[16] contiene L[15] XOR f(R[15],K)
    Il termine PREOUTPUT nello schema indica semplicemente l'unione dei 64 bit
    formata mettendo R[16] a sinistra (bit più significativi) e L[16] a destra (bit meno significativi)
    */
    // 32 bits di shift sono
    uint64_t pre_output =
            ((uint64_t)R_prev << 32) | L_prev; // R[16] va prima convertito a 64 per poterlo shiftare di 32

    // Reverse of initial permutation (IP^-1)
    uint64_t reverse_initial_permutation = permute(pre_output, FP_TABLE, 64, 64);

// copy back into the output buffer
#pragma unroll
    for (int i = 7; i >= 0; i--) {
        output_block[i] = (uint8_t)reverse_initial_permutation; // prendo gli 8 bit meno
                                                                // significativi troncandoli
        reverse_initial_permutation =
                reverse_initial_permutation >> 8; // shifto a destra di 8, distruggo reverse_initial_permutation tanto
                                                  // non mi serve piu'
    }
}