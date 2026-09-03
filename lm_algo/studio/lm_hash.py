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