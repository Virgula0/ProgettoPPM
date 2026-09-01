#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <openssl/evp.h>
#include <openssl/provider.h>

#define MAX_PAD 14
static const unsigned char MAGIC_CONSTANT[8] = {'K', 'G', 'S', '!', '@', '#', '$', '%'};
static const char FULL_CHARSET[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";

static uint8_t NULL_HALF_BYTES[8];
static EVP_CIPHER *DES_CIPHER = NULL;

// ----------------- HELPERS -----------------

/*
 * CONVERSIONE DA 7 BYTE (56 bit) A CHIAVE DES DA 8 BYTE (64 bit)
 * 
 * Il DES richiede in ingresso 64 bit (8 byte), ma la sua sicurezza reale si basa 
 * solo su 56 bit. 
 * 
 * Invece di calcolare la parità per l'8° bit di ogni byte (come richiede lo standard teorico),
 * questo algoritmo impacchetta i 7 byte di input e poi estratti a gruppi di 7 bit,
 * posizionandoli nei 7 bit più significativi di ciascun byte di output.
 * L'8° bit (quello meno significativo, LSB) viene FORZATO A 0 tramite la maschera `& 0xFE`.
 */
void bytes_to_des_key(const uint8_t raw_7_bytes[7], uint8_t key_out[8]) {
    uint64_t b = 0;

    // FASE 1: Impacchettamento
    // Prendiamo i 7 byte di input (7 * 8 = 56 bit) e li concateniamo 
    // in un unico registro a 64 bit `b`.
    for (int i = 0; i < 7; i++) {
        b = (b << 8) | raw_7_bytes[i];
    }
    // Ora `b` contiene i 56 bit del testo nei suoi bit da 0 a 55.

    /*
     * FASE 2: Estrazione dei gruppi da 7 bit e azzeramento dell'8° bit (LSB)
     * 
     * Nota sulla maschera 0xFE:
     * 0xFE in esadecimale equivale a 11111110 in binario.
     * Applicare `& 0xFE` mantiene inalterati i primi 7 bit (da MSB a Bit 1)
     * e FORZA SEMPRE il bit 0 (LSB) a 0.
     */

    // Prende i bit [55..49] -> li sposta in alto e azzera il bit 0
    key_out[0] = (uint8_t)((b >> 49) & 0xFE); 
    
    // Prende i bit [48..42] -> li sposta in alto e azzera il bit 0
    key_out[1] = (uint8_t)((b >> 42) & 0xFE); 
    
    // Prende i bit [41..35] -> li sposta in alto e azzera il bit 0
    key_out[2] = (uint8_t)((b >> 35) & 0xFE); 
    
    // Prende i bit [34..28] -> li sposta in alto e azzera il bit 0
    key_out[3] = (uint8_t)((b >> 28) & 0xFE); 
    
    // Prende i bit [27..21] -> li sposta in alto e azzera il bit 0
    key_out[4] = (uint8_t)((b >> 21) & 0xFE); 
    
    // Prende i bit [20..14] -> li sposta in alto e azzera il bit 0
    key_out[5] = (uint8_t)((b >> 14) & 0xFE); 
    
    // Prende i bit [13..7]  -> li sposta in alto e azzera il bit 0
    key_out[6] = (uint8_t)((b >> 7) & 0xFE);  
    
    // Prende i bit [6..0]   -> li sposta a sinistra di 1 posizione e azzera il bit 0
    key_out[7] = (uint8_t)((b << 1) & 0xFE);  
}

void to_upper(char *str) {
    while (*str) {
        if (*str >= 'a' && *str <= 'z') {
            *str -= ('a' - 'A');
        }
        str++;
    }
}

// Optimized fast DES block encryption using pre-fetched cipher algorithm
void des_encrypt_block(const uint8_t key[8], const uint8_t plaintext[8], uint8_t ciphertext[8]) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int len = 0;

    EVP_EncryptInit_ex2(ctx, DES_CIPHER, key, NULL, NULL);
    EVP_CIPHER_CTX_set_padding(ctx, 0);

    EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, 8);
    EVP_EncryptFinal_ex(ctx, ciphertext + len, &len);

    EVP_CIPHER_CTX_free(ctx);
}

// Convert a hex string to raw bytes
void hex_to_bytes(const char *hex_str, uint8_t *bytes_out, size_t num_bytes) {
    for (size_t i = 0; i < num_bytes; i++) {
        sscanf(hex_str + 2 * i, "%02hhx", &bytes_out[i]);
    }
}

// ----------------- LM HASH CORE -----------------

void hash_half_fast(const char *candidate, uint8_t output_block[8]) {
    uint8_t padded[7] = {0}; // Initialize with zeros
    size_t len = strlen(candidate);

    for (size_t i = 0; i < len && i < 7; i++) {
        char c = candidate[i];
        if (c >= 'a' && c <= 'z') c -= 32;
        padded[i] = (uint8_t)c;
    }

    uint8_t key_bytes[8];
    bytes_to_des_key(padded, key_bytes);
    des_encrypt_block(key_bytes, MAGIC_CONSTANT, output_block);
}

void lm_hash(const char *password, char out_hex[33]) {
    uint8_t padded_pw[14] = {0};
    size_t len = strlen(password);

    for (size_t i = 0; i < len && i < 14; i++) {
        char c = password[i];
        if (c >= 'a' && c <= 'z') c -= 32;
        padded_pw[i] = (uint8_t)c;
    }

    uint8_t key1[8], key2[8];
    bytes_to_des_key(padded_pw, key1);
    bytes_to_des_key(padded_pw + 7, key2);

    uint8_t block1[8], block2[8];
    des_encrypt_block(key1, MAGIC_CONSTANT, block1);
    des_encrypt_block(key2, MAGIC_CONSTANT, block2);

    for (int i = 0; i < 8; i++) sprintf(out_hex + (i * 2), "%02X", block1[i]);
    for (int i = 0; i < 8; i++) sprintf(out_hex + 16 + (i * 2), "%02X", block2[i]);
    out_hex[32] = '\0';
}

// ----------------- BRUTE FORCE ENGINE -----------------

int build_combo_dfs(char *current, int depth, int target_depth, const uint8_t target_bytes[8], char *found_match) {
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

int crack_half(const uint8_t target_bytes[8], char *cracked_out) {
    if (memcmp(target_bytes, NULL_HALF_BYTES, 8) == 0) {
        cracked_out[0] = '\0';
        return 1;
    }

    char buffer[8];
    for (int len = 1; len <= 7; len++) {
        printf("[*] Testing length %d...\n", len);
        if (build_combo_dfs(buffer, 0, len, target_bytes, cracked_out)) {
            return 1;
        }
    }
    cracked_out[0] = '\0';
    return 0;
}

void crack_hash(const char *hash_hex, char cracked_password[15]) {
    uint8_t target1[8], target2[8];
    hex_to_bytes(hash_hex, target1, 8);
    hex_to_bytes(hash_hex + 16, target2, 8);

    char part1[8] = {0}, part2[8] = {0};

    printf("[+] Cracking first half...\n");
    crack_half(target1, part1);

    printf("[+] Cracking second half...\n");
    crack_half(target2, part2);

    snprintf(cracked_password, 15, "%s%s", part1, part2);
}

// ----------------- MAIN -----------------

int main(void) {
    // OpenSSL 3.0+ setup
    OSSL_PROVIDER_load(NULL, "legacy");
    OSSL_PROVIDER_load(NULL, "default");
    DES_CIPHER = EVP_CIPHER_fetch(NULL, "DES-ECB", NULL);

    // Compute NULL_HALF_BYTES once globally (Optimization 1)
    uint8_t null_raw[7] = {0};
    uint8_t null_key[8];
    bytes_to_des_key(null_raw, null_key);
    des_encrypt_block(null_key, MAGIC_CONSTANT, NULL_HALF_BYTES);

    const char *target_password = "abc";
    char hash_val[33];
    lm_hash(target_password, hash_val);

    printf("Password : %s\n", target_password);
    printf("LM Hash  : %s\n\n", hash_val);

    clock_t start = clock();

    char cracked_password[15];
    crack_hash(hash_val, cracked_password);

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\n[!] Success! Cracked Password: %s\n", cracked_password);
    printf("Taken %.4f seconds\n", time_taken);

    EVP_CIPHER_free(DES_CIPHER);
    return 0;
}
