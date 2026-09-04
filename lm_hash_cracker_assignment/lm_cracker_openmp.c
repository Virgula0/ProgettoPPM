#include <openssl/evp.h>
#include <openssl/provider.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <stdbool.h>

#define MAX_PAD 14
static const unsigned char MAGIC_CONSTANT[8] = {'K', 'G', 'S', '!', '@', '#', '$', '%'};
static const char FULL_CHARSET[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";

static uint8_t NULL_HALF_BYTES[8];
static EVP_CIPHER* DES_CIPHER = NULL;

// ----------------- HELPERS -----------------

void bytes_to_des_key(const uint8_t raw_7_bytes[7], uint8_t key_out[8]) {
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

void to_upper(char* str) {
    while (*str) {
        if (*str >= 'a' && *str <= 'z') {
            *str -= ('a' - 'A');
        }
        str++;
    }
}

// Optimized fast DES block encryption using pre-fetched cipher algorithm
void des_encrypt_block(const uint8_t key[8], const uint8_t plaintext[8], uint8_t ciphertext[8]) {
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    int len = 0;

    EVP_EncryptInit_ex2(ctx, DES_CIPHER, key, NULL, NULL);
    EVP_CIPHER_CTX_set_padding(ctx, 0);

    EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, 8);
    EVP_EncryptFinal_ex(ctx, ciphertext + len, &len);

    EVP_CIPHER_CTX_free(ctx);
}

// Convert a hex string to raw bytes
void hex_to_bytes(const char* hex_str, uint8_t* bytes_out, size_t num_bytes) {
    for (size_t i = 0; i < num_bytes; i++) {
        sscanf(hex_str + 2 * i, "%02hhx", &bytes_out[i]);
    }
}

// ----------------- LM HASH CORE -----------------

void hash_half_fast(const char* candidate, uint8_t output_block[8]) {
    uint8_t padded[7] = {0}; // Initialize with zeros
    size_t len = strlen(candidate);

    for (size_t i = 0; i < len && i < 7; i++) {
        char c = candidate[i];
        if (c >= 'a' && c <= 'z')
            c -= 32;
        padded[i] = (uint8_t)c;
    }

    uint8_t key_bytes[8];
    bytes_to_des_key(padded, key_bytes);
    des_encrypt_block(key_bytes, MAGIC_CONSTANT, output_block);
}

void lm_hash(const char* password, char out_hex[33]) {
    uint8_t padded_pw[14] = {0};
    size_t len = strlen(password);

    for (size_t i = 0; i < len && i < 14; i++) {
        char c = password[i];
        if (c >= 'a' && c <= 'z')
            c -= 32;
        padded_pw[i] = (uint8_t)c;
    }

    uint8_t key1[8], key2[8];
    bytes_to_des_key(padded_pw, key1);
    bytes_to_des_key(padded_pw + 7, key2);

    uint8_t block1[8], block2[8];
    des_encrypt_block(key1, MAGIC_CONSTANT, block1);
    des_encrypt_block(key2, MAGIC_CONSTANT, block2);

    for (int i = 0; i < 8; i++)
        sprintf(out_hex + (i * 2), "%02X", block1[i]);
    for (int i = 0; i < 8; i++)
        sprintf(out_hex + 16 + (i * 2), "%02X", block2[i]);
    out_hex[32] = '\0';
}

// ----------------- BRUTE FORCE ENGINE -----------------

int build_combo_dfs(char* current, int depth, int target_depth, const uint8_t target_bytes[8], char* found_match,
                    volatile int* found) {
    if (*found)
        return 0;

    if (depth == target_depth) {
        current[depth] = '\0';
        uint8_t candidate_block[8];
        hash_half_fast(current, candidate_block);

        if (memcmp(candidate_block, target_bytes, 8) == 0) {
            strcpy(found_match, current);
            *found = 1;
            return 1;
        }
        return 0;
    }

    size_t charset_len = strlen(FULL_CHARSET);
    for (size_t i = 0; i < charset_len; i++) {
        if (*found)
            break;
        current[depth] = FULL_CHARSET[i];
        if (build_combo_dfs(current, depth + 1, target_depth, target_bytes, found_match, found)) {
            return 1;
        }
    }
    return 0;
}

int crack_half(const uint8_t target_bytes[8], char* cracked_out) {
    if (memcmp(target_bytes, NULL_HALF_BYTES, 8) == 0) {
        cracked_out[0] = '\0';
        return 1;
    }

    volatile int found = 0;
    size_t charset_len = strlen(FULL_CHARSET);

    for (int len = 1; len <= 7; len++) {
        printf("[*] Testing length %d...\n", len);

#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < charset_len; i++) {
            if (found)
                continue;

            char buffer[8] = {0};
            buffer[0] = FULL_CHARSET[i];

            if (len == 1) {
                uint8_t candidate_block[8];
                hash_half_fast(buffer, candidate_block);

                if (memcmp(candidate_block, target_bytes, 8) == 0) {
                    strcpy(cracked_out, buffer);
                    found = 1;
                }
            } else {
                build_combo_dfs(buffer, 1, len, target_bytes, cracked_out, &found);
            }
        }

        if (found)
            return 1;
    }
    cracked_out[0] = '\0';
    return 0;
}

void crack_hash(const char* hash_hex, char cracked_password[15]) {
    uint8_t target1[8], target2[8];
    hex_to_bytes(hash_hex, target1, 8);
    hex_to_bytes(hash_hex + 16, target2, 8);

    char part1[8] = {0}, part2[8] = {0};

#pragma omp parallel sections
    {
#pragma omp section
        {
            printf("[+] Cracking first half...\n");
            crack_half(target1, part1);
        }
#pragma omp section
        {
            printf("[+] Cracking second half...\n");
            crack_half(target2, part2);
        }
    }

    snprintf(cracked_password, 15, "%s%s", part1, part2);
}

void toUpper(char* str) {
    while (*str) {
        if (*str >= 'a' && *str <= 'z') {
            *str -= ('a' - 'A');
        }
        str++;
    }
}

bool checkHEXRange(char ch) {
    return (ch >= 0x30 && ch <= 0x39) || (ch >= 0x41 && ch <= 0x46) || (ch >= 0x61 && ch <= 0x66);
}

bool checkValidHash(char* toCrack) {
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

double elapsed_seconds(struct timespec start, struct timespec end) {
    // calcola il wall-clock time, CPU + GPU overhead, tempo reale ed effetttivo impiegato
    // dall'algoritmo senza trucchi
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

// ----------------- MAIN -----------------

int main(int argc, char** argv) {
    omp_set_max_active_levels(2);

    if (argc < 2) {
        printf("[ERR] No cracking hash provided\n");
        return -1;
    }

    char* hash_val = argv[1];

    if (!checkValidHash(hash_val)) {
        printf("[ERR] Provided hash seem a not valid LM HASH\n");
        return -1;
    }

    toUpper(hash_val);

    // OpenSSL 3.0+ setup
    OSSL_PROVIDER_load(NULL, "legacy");
    OSSL_PROVIDER_load(NULL, "default");
    DES_CIPHER = EVP_CIPHER_fetch(NULL, "DES-ECB", NULL);

    // Compute NULL_HALF_BYTES once globally (Optimization 1)
    uint8_t null_raw[7] = {0};
    uint8_t null_key[8];
    bytes_to_des_key(null_raw, null_key);
    des_encrypt_block(null_key, MAGIC_CONSTANT, NULL_HALF_BYTES);

    struct timespec startTime, endTime;
    clock_gettime(CLOCK_MONOTONIC, &startTime); // CLOCK_MONOTONIC  non e' soggetto al datetime locale, tempo puro

    char cracked_password[15];
    crack_hash(hash_val, cracked_password);

    clock_gettime(CLOCK_MONOTONIC, &endTime);
    double time_taken = elapsed_seconds(startTime, endTime);

    printf("\n[!] Success! Cracked Password: %s\n", cracked_password);
    printf("Taken %.4f seconds\n", time_taken);

    EVP_CIPHER_free(DES_CIPHER);
    return 0;
}