from Crypto.Cipher import DES
import string

class LMHash:
    max_pad = 14
    null_byte_charset = 0x00
    full_charset = string.ascii_uppercase + string.digits + string.punctuation + string.whitespace

    def __init__(self):
        pass

    def __parity_bit(self, a):
        ones_count = a.count("1")
        parity_bit = "1" if (ones_count % 2 == 0) else "0"
        to_pad = 7 - len(a)
        return "".join(["0" for _ in range(to_pad)]) + a + parity_bit

    def __bytes_to_bit(self, decimal_to_convert, accumulated=""):
        accumulated = str(decimal_to_convert % 2) + accumulated
        to_pad = 8 - len(accumulated)

        if decimal_to_convert < 2:
            return "".join(["0" for _ in range(to_pad)]) + accumulated
        else:
            return self.__bytes_to_bit(int(decimal_to_convert / 2), accumulated)

    def __bit_stream_parity(self, bytes_to_convert):
        full_56_bits = ""
        for current in bytes_to_convert:
            full_56_bits += self.__bytes_to_bit(current, "")

        bit_array_with_parity = []
        for i in range(0, 56, 7):
            group_7 = full_56_bits[i : i + 7]
            bits_with_parity = self.__parity_bit(group_7)
            bit_array_with_parity.extend(int(x) for x in list(bits_with_parity))

        return bit_array_with_parity

    def __bits_to_bytes(self, bit_list):
        """Prende una lista di 64 int (0/1) e la riconverte in 8 byte puri."""
        byte_array = bytearray()
        # Raggruppa i bit 8 a 8
        for i in range(0, 64, 8):
            byte_bits = bit_list[i : i + 8]
            # Trasforma i bit in una stringa '10101001' e poi in intero/byte
            byte_str = "".join(str(b) for b in byte_bits)
            byte_array.append(int(byte_str, 2))
        return bytes(byte_array)

    def hash(self, to_encrypt):
        if len(to_encrypt) > 14:
            raise ValueError("password must be less than 14 chars")

        # 1. Prepariamo i byte della password con padding a 14 byte
        enc_bytes = bytearray(to_encrypt.upper().encode("CP437"))
        to_pad = self.max_pad - len(enc_bytes)
        enc_bytes += bytes([0x00 for _ in range(to_pad)])

        # 2. Split nelle due metà da 7 byte
        l = int(len(enc_bytes) / 2)
        first_half = enc_bytes[:l]
        second_half = enc_bytes[l:]

        # 3. Generazione bit stream a 64 bit (liste di 64 interi 0/1)
        bit_stream_first_half = self.__bit_stream_parity(first_half)
        bit_stream_second_half = self.__bit_stream_parity(second_half)

        # 4. Conversione delle liste di bit in byte per PyCryptodome (8 byte ciascuna)
        key_1_bytes = self.__bits_to_bytes(bit_stream_first_half)
        key_2_bytes = self.__bits_to_bytes(bit_stream_second_half)

        # 5. La costante fissa di Microsoft
        magic_constant = b"KGS!@#$%"

        # 6. Cifratura DES con PyCryptodome in modalità ECB
        cipher1 = DES.new(key_1_bytes, DES.MODE_ECB)
        cipher2 = DES.new(key_2_bytes, DES.MODE_ECB)

        block1 = cipher1.encrypt(magic_constant)
        block2 = cipher2.encrypt(magic_constant)

        # 7. Unione dei due blocchi cifrati da 8 byte -> 16 byte totali (32 char HEX)
        lm_hash = (block1 + block2).hex().upper()
        return lm_hash

    def _hash_for_cracker_optimized_on_halves(self, to_encrypt):
        if len(to_encrypt) > 7:
            raise ValueError("password must be less than 7 chars")

        # 1. Prepariamo i byte della password con padding a 7 byte
        enc_bytes = bytearray(to_encrypt.upper().encode("CP437"))
        to_pad = 7 - len(enc_bytes)
        enc_bytes += bytes([0x00 for _ in range(to_pad)])

        # 3. Generazione bit stream a 64 bit (liste di 64 interi 0/1)
        bit_stream_half = self.__bit_stream_parity(enc_bytes)

        # 4. Conversione delle liste di bit in byte per PyCryptodome (8 byte ciascuna)
        key_bytes = self.__bits_to_bytes(bit_stream_half)

        # 5. La costante fissa di Microsoft
        magic_constant = b"KGS!@#$%"

        # 6. Cifratura DES con PyCryptodome in modalità ECB
        cipher1 = DES.new(key_bytes, DES.MODE_ECB)

        block1 = cipher1.encrypt(magic_constant)
        return block1.hex().upper()

    def is_null(self, hashToCheck):
        enc_bytes = bytearray([0x00 for _ in range(0,8)])
        bit_stream = self.__bit_stream_parity(enc_bytes)
        key = self.__bits_to_bytes(bit_stream)
        magic_constant = b"KGS!@#$%"
        cipher1 = DES.new(key, DES.MODE_ECB)
        block = cipher1.encrypt(magic_constant)
        print("NULL BYTES ENCRYPTED: ", " ".join([hex(x) for x in block]))
        return block.hex().upper() == hashToCheck.upper()

    def _combinations_iterative(self, length: int) -> list[str]:
        if length <= 0:
            return []

        result = [""]
        for _ in range(length):
            next_result = []
            for combo in result:
                for char in self.full_charset:
                    next_result.append(combo + char)
            result = next_result

        return result

    def _crack_half(self, half_psw):
        # HALF BRUTEFORCE
        for len_to_bruteforce in range(1,8): # from 1 to 7
            # each len_to_bruteforce must iterate over the full_charset
            # needs to apply hash function to each new candidate and check against the half
            print("Reached length " + str(len_to_bruteforce))
            candidates = self._combinations_iterative(len_to_bruteforce)
            for i in candidates:
                to_check = self._hash_for_cracker_optimized_on_halves(i) # first half 16 bytes already
                if to_check == half_psw:
                    return i

    def crack(self, hashToCrack: str):
        #split in two halves
        if len(hashToCrack) != 32:
            raise ValueError("Length of hash not matches")

        first_half = (hashToCrack[:len(hashToCrack)//2]).upper()
        second_half = hashToCrack[len(hashToCrack)//2:].upper()

        # first check if one of the halves or both is just a bunch of null padded bytes
        is_first_half_nulled = self.is_null(first_half)

        if is_first_half_nulled:
            raise ValueError("hash is just 0x00 repeated 14 times")

        is_second_half_nulled = self.is_null(second_half)
        cracked_first_half = self._crack_half(first_half)

        if is_second_half_nulled:
            return cracked_first_half

        cracked_second_half = self._crack_half(second_half)

        return cracked_first_half + cracked_second_half


import time

lm_hash = LMHash()
we = lm_hash.hash("abc")
print(we)
before = time.time()
print("Cracked! " + lm_hash.crack(we))
print(f"Taken {time.time()-before}")

"""
Cuda like schema:

[Host CPU]
   │
   ├── 1. Load Target Hash (e.g., 32 hex chars -> split into two 8-byte target outputs)
   ├── 2. Generate Search Space Offsets (e.g., Key Range 0 to 94,931,877,132)
   └── 3. Dispatch to GPU Kernel (grid of Blocks x Threads)
             │
             ├── Thread 0       ---> Try Key Index N     ---> Compute DES("KGS!@#$%") ---> Compare target
             ├── Thread 1       ---> Try Key Index N+1   ---> Compute DES("KGS!@#$%") ---> Compare target
             ├── Thread 2       ---> Try Key Index N+2   ---> Compute DES("KGS!@#$%") ---> Compare target
             └── Thread (N)     ---> ...
                                         │
                                         └── Match Found? Write cleartext to global flag memory & exit.


Total combinations:

ignorando le lettere minuscole che sono convertite in uppercase: ?u + ?d + ?s sono in totale = 26+10+33 = 69

Per ciascuna meta': 69^7 = 7446353252589

Totale = 2*Meta' = 14892706505178

Approccio Da usare (2 kernel in parallelo tramite gli streams)

lancio operazioni in parallelo tramite gli streams:

#include <cuda_runtime.h>
#include <iostream>

// Funzione helper per configurare ed eseguire i due stream
void launch_lm_cracker(uint64_t total_combinations, const uint8_t* target_half1, const uint8_t* target_half2) {
    int threadsPerBlock = 0;
    int minGridSize = 0;

    // 1. Chiedi a CUDA la configurazione OTTIMALE per il tuo kernel sulla GPU in uso
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,       // Minimo numero di blocchi per saturare la GPU
        &threadsPerBlock,   // Numero di thread per blocco suggerito (es. 256 o 512)
        crack_lm_half,      // Nome della funzione __global__ del kernel
        0,                  // Memoria dinamica shared (0 se non ne usi)
        0                   // Limite max di blocchi (0 = nessun limite)
    );

    // 2. Calcola i blocchi totali per coprire la scheda video.
    // Moltiplicare minGridSize (es. per 2 o 4) garantisce che la GPU rimanga piena
    // anche quando un insieme di blocchi termina prima di altri.
    int blocksPerGrid = minGridSize * 4;

    std::cout << "Configurazione dinamica CUDA:" << std::endl;
    std::cout << " - Thread per Blocco: " << threadsPerBlock << std::endl;
    std::cout << " - Blocchi per Grid: " << blocksPerGrid << std::endl;
    std::cout << " - Thread totali per Stream: " << (blocksPerGrid * threadsPerBlock) << std::endl;

    // 3. Creazione di 2 CUDA Stream indipendenti
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // 4. Allocazione e copia della memoria GPU per le due metà...
    // (target_gpu1, target_gpu2, result_gpu1, result_gpu2, ecc.)

    // 5. Lancio asincrono sui due stream
    // ENTRAMBI usano la configurazione ottimale (blocksPerGrid, threadsPerBlock)
    crack_lm_half<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
        0, total_combinations, target_gpu1, result_gpu1, flag_gpu1
    );

    crack_lm_half<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(
        0, total_combinations, target_gpu2, result_gpu2, flag_gpu2
    );

    // 6. Attendi la conclusione di entrambi gli stream
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Pulizia stream
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

3. Perché NON dividere per 2 per i due stream?

Gestione dell'Hardware (Hardware Scheduler): La GPU possiede un componente hardware chiamato GigaThread Engine.
Quando lanci due kernel su due stream diversi, la GPU distribuisce i blocchi del Stream 1 e dello Stream 2 sui vari multiprocessori (SM) disponibili.

$$\text{totalThreads} = \text{blocksPerGrid} \times \text{threadsPerBlock}$$

cudaOccupancyMaxPotentialBlockSize -> che interroga la scheda video installata e calcola dinamicamente la combinazione perfetta di blocchi e
thread per ottenere il 100% delle prestazioni dal tuo kernel.

"""

"""
================================================================================
                    ARCHITETTURA LM HASH CRACKER (CUDA)
================================================================================

1. CHIARIMENTO SUL NUMERO DI COMBINAZIONI (KEY RANGE)
--------------------------------------------------------------------------------
La differenza nel numero di combinazioni per ciascuna metà di 7 byte dipende
esclusivamente dalla dimensione del charset considerato (N):

- Charset Alfanumerico Ridotto (A-Z, 0-9) [N = 37]:
    Combinazioni = 37^7 = 94.931.877.132 (~94,9 miliardi)
    * Questo è il valore che avevi letto originariamente.

- Charset Standard Hashcat (?u + ?d + ?s) [N = 69]:
    Maiuscole (26) + Cifre (10) + Speciali (33) = 69 caratteri totali.
    Combinazioni = 69^7 = 7.537.088.669.173 (~7,53 trilioni)
    (Sommando lunghezze 1..7: sum(69^i) = 7.646.593.864.063)

Poiché le due metà da 7 byte sono completamente indipendenti (cifrate separatamente
con DES-ECB sul blocco fisso "KGS!@#$%"), il carico totale è ADDITIVO, non moltiplicativo:
    Totale Lavoro = (Combinazioni Metà 1) + (Combinazioni Metà 2)


2. ARCHITETTURA STREAM CUDA (PARALLELISMO TRA LE DUE METÀ)
--------------------------------------------------------------------------------
- Unico Kernel Parametrizzato: `crack_lm_half` gestisce l'analisi di una singola metà.
- Due CUDA Streams Asincroni:
    * Stream 1: elabora la prima metà dell'hash target.
    * Stream 2: elabora la seconda metà dell'hash target.
- Configurazione Grid/Block:
    * NON si divide la dimensione della Grid a metà!
    * Si calcola la saturazione ottimale della GPU (tramite `cudaOccupancyMaxPotentialBlockSize`)
      e si applica la stessa configurazione completa a ENTRAMBI gli stream.
    * Il GigaThread Engine della GPU distribuirà dinamicamente i blocchi di entrambi
      gli stream sugli SM (Streaming Multiprocessors).
    * Vantaggio Early-Exit: Se la password è <= 7 caratteri, la 2ª metà (solo padding 0x00)
      finisce all'istante e la GPU assegna il 100% delle risorse allo Stream 1.


3. ORGANIZZAZIONE INTERNA DEL LAVORO NEL SINGOLO KERNEL
--------------------------------------------------------------------------------
Ogni thread processa l'INTERA candidate key di 7 caratteri alla volta.

a) Grid-Stride Loop (Nessuna sovrapposizione tra thread):
    - Sia `tid` l'ID globale del thread (`blockIdx.x * blockDim.x + threadIdx.x`).
    - Sia `stride` la dimensione totale della Grid (`gridDim.x * blockDim.x`).
    - Il loop avanza deterministicamente: `for (uint64_t idx = tid; idx < total_keys; idx += stride)`
      * Thread 0  -> esamina indici: 0, 0+stride, 0+2*stride, ...
      * Thread 1  -> esamina indici: 1, 1+stride, 1+2*stride, ...
      * Nessun thread sovrappone mai il proprio lavoro con un altro.

b) Conversione Base-N (Indice Numerico -> Stringa Candidate Key):
    - L'indice 64-bit del thread (`idx`) viene convertito in una stringa di 7 caratteri
      tramite operatore modulo e divisione progressiva per la dimensione del charset (N):
      `char_pos_i = charset[(idx / N^i) % N]`

c) Pipeline di Esecuzione del Thread:
    1. Genera candidate key a 7 caratteri dall'indice `idx`.
    2. Espande i 7 byte (56 bit) aggiungendo i bit di parità DES -> 8 byte (64 bit).
    3. Cifra la costante "KGS!@#$%" usando la chiave DES creata.
    4. Confronta l'output cifrato con la metà target assegnata allo stream.
    5. In caso di match, salva il risultato in memoria globale e attiva il flag
       atomico di stop per interrompere gli altri thread.
================================================================================
"""