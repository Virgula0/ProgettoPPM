import time
import mmh3

K_HASHES = 6  # Numero di funzioni hash


def load_passwords(filename: str, max_lines: int = 0) -> list[str]:
    """Carica le password da un file .txt gestendo terminatori di riga Windows/Linux."""
    passwords = []
    try:
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if line:
                    passwords.append(line)
                    if max_lines > 0 and len(passwords) >= max_lines:
                        break
    except FileNotFoundError:
        print(f"Errore: impossibile aprire il file '{filename}'!")
    return passwords


class BloomFilterSeq:
    """Bloom Filter Sequenziale."""

    def __init__(self, size: int):
        self.size = size
        self.bit_array = bytearray(size)

    def _hash(self, item: str) -> tuple[int, int]:
        """Calcola l'hash MurmurHash3 a 128 bit e restituisce la coppia {h1, h2}."""
        val = mmh3.hash128(item, 0, signed=False)
        h1 = val & 0xFFFFFFFFFFFFFFFF
        h2 = (val >> 64) & 0xFFFFFFFFFFFFFFFF
        return h1, h2

    def _hash_single(self, h1: int, h2: int, hash_idx: int) -> int:
        """Combina h1 e h2 con l'indice tramite Double Hashing."""
        return (h1 + hash_idx * h2) % self.size

    def add(self, item: str):
        """Inserisce un singolo elemento nel Bloom Filter."""
        h1, h2 = self._hash(item)
        for j in range(K_HASHES):
            idx = self._hash_single(h1, h2, j)
            self.bit_array[idx] = 1

    def contains(self, item: str) -> bool:
        """Verifica se un singolo elemento è presente nel Bloom Filter."""
        h1, h2 = self._hash(item)
        for j in range(K_HASHES):
            idx = self._hash_single(h1, h2, j)
            if self.bit_array[idx] == 0:
                return False
        return True

    def add_from_file(self, items: list[str]):
        """Inserisce una lista di elementi nel Bloom Filter (sequenziale)."""
        for item in items:
            self.add(item)

    def contains_from_file(self, items: list[str]) -> int:
        """Conta quanti elementi della lista sono presenti nel Bloom Filter (sequenziale)."""
        count = 0
        for item in items:
            if self.contains(item):
                count += 1
        return count


def main():
    filename = "rockyou.txt"
    ctrl_filename = "parole_uniche.txt"

    filter_size = 143_443_800  # Dimensione dell'array di bit (circa 17 MB)
    num_cycles = 15

    print(f"Caricamento password da '{filename}'...")
    passwords = load_passwords(filename, 0)
    if not passwords:
        print("Caricamento fallito, l'array è vuoto.")
        return
    print(f"Caricate {len(passwords)} password.\n")

    print(f"Caricamento password da '{ctrl_filename}'...")
    ctrl_passwords = load_passwords(ctrl_filename, 0)
    if not ctrl_passwords:
        print("Caricamento fallito, l'array di controllo è vuoto.")
        return
    print(f"Caricate {len(ctrl_passwords)} password.\n")

    tot_add_time_seq = 0.0
    tot_srch_time_seq = 0.0

    for i in range(num_cycles):
        bloom_seq = BloomFilterSeq(filter_size)

        # ---------------- OPERAZIONE ADD ----------------
        start_add_seq = time.perf_counter()
        start_add_seq_cpu = time.process_time()

        bloom_seq.add_from_file(passwords)

        time_add_seq = time.perf_counter() - start_add_seq
        time_add_seq_cpu = time.process_time() - start_add_seq_cpu

        # ---------------- OPERAZIONE CONTAINS ----------------
        start_srch_seq = time.perf_counter()
        start_srch_seq_cpu = time.process_time()

        res_seq = bloom_seq.contains_from_file(ctrl_passwords)

        time_srch_seq = time.perf_counter() - start_srch_seq
        time_srch_seq_cpu = time.process_time() - start_srch_seq_cpu

        # ---------------- STAMPA RISULTATI CICLO ----------------
        print(f"CICLO SPERIMENTALE NUMERO {i + 1}\n")

        print("=== INSERIMENTO (ADD) ===")
        print(f"Tempo Sequenziale:     {time_add_seq:.6f} s")
        print(f"Tempo CPU Sequenziale: {time_add_seq_cpu:.6f} s\n")

        print("=== RICERCA (CONTAINS) ===")
        print(f"Tempo Sequenziale:     {time_srch_seq:.6f} s")
        print(f"Tempo CPU Sequenziale: {time_srch_seq_cpu:.6f} s")
        print(f"Elementi trovati:      {res_seq}\n")

        tot_add_time_seq += time_add_seq
        tot_srch_time_seq += time_srch_seq

    # ---------------- MEDIE FINALI ----------------
    print("=== RISULTATI FINALI ADD (MEDIA) ===")
    print(f"Tempo Medio Sequenziale: {tot_add_time_seq / num_cycles:.6f} s\n")

    print("=== RISULTATI FINALI CONTAINS (MEDIA) ===")
    print(f"Tempo Medio Sequenziale: {tot_srch_time_seq / num_cycles:.6f} s\n")


if __name__ == "__main__":
    main()