import os
import sys
import time
import mmh3
import threading
from multiprocessing.pool import ThreadPool

# NUMBER_OF_THREADS = os.cpu_count()
NUMBER_OF_THREADS = os.cpu_count() or 8

"""
###################################################
    OMP_PROC_BIND="spread"
    OMP_PLACES=threads
###################################################
"""

K_HASHES = 6  # numero di funzioni hash

# numero di lock usati per lo striping sul bit_array condiviso: piu' lock = meno contesa
# ma anche piu' overhead di gestione. 4096 e' un compromesso ragionevole: abbastanza
# granulare da non serializzare troppo i thread, ma non cosi' tanto da rendere il costo
# di allocazione/acquisizione dei lock stessi un collo di bottiglia
NUM_LOCK_STRIPES = 4096

# dimensione del chunk dinamico assegnato ad ogni thread quando richiede nuovo lavoro,
# equivalente a "schedule(dynamic, 1024)" in OpenMP
DYNAMIC_CHUNK_SIZE = 1024


# funzione di supporto per caricare le password da file .txt
def load_passwords(filename: str, max_lines: int = 0) -> list[str]:
    passwords = []
    try:
        with open(filename, "r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                if line:
                    line = line.rstrip("\n")
                    if line:
                        # Rimuove l'eventuale carattere '\r' da file formattati in windows
                        if line.endswith("\r"):
                            line = line[:-1]
                        passwords.append(line)
                        if max_lines > 0 and len(passwords) >= max_lines:
                            break
    except Exception:
        print(f"Errore: impossibile aprire il file '{filename}'!")
        return passwords
    return passwords


"""
####################################################################################################
    BLOOM FILTER ASTRATTO
####################################################################################################
"""


class BloomFilter:
    def __init__(self, size: int):
        # dimensione logica del bit_array, cioe' il numero di bit indirizzabili (non il numero di byte
        # allocati)
        self.size = size

        # bit-packing: invece di un uint8_t per ogni singolo bit (1 byte = 1 bit usato, 7 sprecati),
        # impacchettiamo 8 bit logici in ogni singolo byte fisico. Questo riduce l'allocazione di un fattore 8x
        # (es. 143443800 bit -> ~17.9MB invece di ~137MB), e come bonus riduce anche il traffico di memoria
        # verso la RAM, perche' piu' bit "utili" stanno nella stessa cache line da 64 byte.
        # il costruttore continua a ricevere "size" come numero di BIT desiderati (stessa interfaccia di prima,
        # cosi' il calcolo di filter_size nel main non deve cambiare), ma internamente alloca solo size/8 byte
        self.bit_array = bytearray((size + 7) // 8)

    # distruttore
    def __del__(self):
        pass

    # Calcola l'hash a 128 bit una sola volta e restituisce la coppia {h1, h2}
    def _hash(self, data: bytes) -> tuple[int, int]:
        hash_128 = mmh3.hash128(data, 0, signed=False)
        h1 = hash_128 & 0xFFFFFFFFFFFFFFFF
        h2 = (hash_128 >> 64) & 0xFFFFFFFFFFFFFFFF
        return h1, h2

    # Combina h1 e h2 con l'indice (Double Hashing) restituendo solo l'indice calcolato
    # questo resta un indice di bit logico (0..size-1), non un indice di byte:
    # la conversione bit -> byte/offset avviene solo nei metodi che accedono a bit_array
    def _hash_single(self, h1: int, h2: int, hash_idx: int) -> int:
        return (h1 + hash_idx * h2) % self.size

    # funzioni di supporto per settare/leggere un singolo bit dato il suo indice logico
    # bit_index / 8  -> in che byte si trova
    # bit_index % 8  -> in che posizione dentro quel byte si trova
    # versione non atomica, usata dalla parte sequenziale
    def _set_bit(self, bit_index: int) -> None:
        self.bit_array[bit_index >> 3] |= 1 << (bit_index & 7)

    def _get_bit(self, bit_index: int) -> bool:
        return (self.bit_array[bit_index >> 3] & (1 << (bit_index & 7))) != 0

    # metodi di default
    def add(self, item: str) -> None:
        item_bytes = item.encode("utf-8")
        h1, h2 = self._hash(item_bytes)
        for j in range(K_HASHES):
            self._set_bit(self._hash_single(h1, h2, j))

    def contains(self, item: str) -> bool:
        item_bytes = item.encode("utf-8")
        h1, h2 = self._hash(item_bytes)
        for j in range(K_HASHES):
            if not self._get_bit(self._hash_single(h1, h2, j)):
                return False
        return True

    # metodi virtuali
    def add_from_file(self, items: list[str]) -> None:
        raise NotImplementedError

    def contains_from_file(self, items: list[str]) -> int:
        raise NotImplementedError


"""
####################################################################################################
    BLOOM FILTER PARALLELO
####################################################################################################
"""


class BloomFilterPar(BloomFilter):
    def __init__(self, size: int):
        super().__init__(size)
        # array di lock per lo striping: ogni lock protegge un sottoinsieme di byte del
        # bit_array condiviso (byte_index % NUM_LOCK_STRIPES). Questo e' l'equivalente
        # "a grana grossa" del #pragma omp atomic update del C++: la' l'hardware garantisce
        # l'atomicita' della singola read-modify-write sul byte, qui non abbiamo un'istruzione
        # atomica a livello di bytearray Python, quindi dobbiamo simularla con un lock. Usare
        # un solo lock globale funzionerebbe ma serializzerebbe completamente l'add (nessun
        # parallelismo reale); usarne molti (stripe) riduce la probabilita' che due thread
        # vogliano lo stesso lock nello stesso momento, avvicinandosi al comportamento della
        # versione C++ dove la contesa e' limitata al singolo byte
        self._stripe_locks = [threading.Lock() for _ in range(NUM_LOCK_STRIPES)]

        # contatore condiviso + lock dedicato per l'assegnazione dinamica dei chunk di lavoro,
        # equivalente a "schedule(dynamic, 1024)" in OpenMP: ogni thread, appena finisce il suo
        # chunk, ne richiede uno nuovo invece di ricevere una porzione fissa e uguale per tutti
        # (schedule statico), cosi' il lavoro si bilancia meglio se gli elementi non sono
        # uniformi in costo di hashing/scrittura
        self._next_index = 0
        self._sched_lock = threading.Lock()

    # restituisce il prossimo chunk (start, end) da processare, oppure None se il lavoro e' finito.
    # analogo al meccanismo interno con cui il runtime OpenMP distribuisce gli indici del ciclo
    # ai thread quando si usa schedule(dynamic, ...)
    def _get_next_chunk(self, total_items: int) -> tuple[int, int] | None:
        with self._sched_lock:
            start = self._next_index
            if start >= total_items:
                return None
            end = min(start + DYNAMIC_CHUNK_SIZE, total_items)
            self._next_index = end
            return start, end

    def add_from_file(self, items: list[str]) -> None:
        # reset del contatore di scheduling dinamico ad ogni chiamata, cosi' l'oggetto e'
        # riutilizzabile su piu' cicli sperimentali senza ricrearlo
        self._next_index = 0
        total_items = len(items)

        def worker():
            while True:
                chunk = self._get_next_chunk(total_items)
                if chunk is None:
                    break
                start, end = chunk
                for i in range(start, end):
                    item_bytes = items[i].encode("utf-8")
                    h1, h2 = self._hash(item_bytes)
                    for j in range(K_HASHES):
                        bit_index = self._hash_single(h1, h2, j)
                        # bit-packing + parallelismo: con il vecchio schema (1 byte per bit), due
                        # thread che scrivevano bit diversi scrivevano anche BYTE diversi, quindi
                        # l'operazione era gia' di per se' "safe" a livello di singolo elemento
                        # (anche se soffriva di false sharing sulla cache line). Ora che 8 bit
                        # logici condividono lo stesso byte fisico, se due thread impostano due
                        # bit diversi ma nello stesso byte con una normale "|=" si crea una vera
                        # race condition (read-modify-write non atomico), con rischio concreto di
                        # perdere aggiornamenti. Serve quindi rendere atomica l'operazione di OR
                        # sul byte: qui lo facciamo prendendo il lock dello stripe corrispondente
                        byte_index = bit_index >> 3
                        mask = 1 << (bit_index & 7)
                        lock = self._stripe_locks[byte_index % NUM_LOCK_STRIPES]
                        with lock:
                            self.bit_array[
                                byte_index
                            ] |= mask  # concateno i bit della maschera al bit_array

        threads = [threading.Thread(target=worker) for _ in range(NUMBER_OF_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # ricerca di un batch di password nel Bloom Filter
    # la lettura resta senza lock: piu' thread che leggono lo stesso byte in contemporanea non
    # creano race condition (nessuno modifica il dato durante la ricerca)
    def contains_from_file(self, items: list[str]) -> int:
        chunk_size = DYNAMIC_CHUNK_SIZE

        def process_chunk(chunk):
            count = 0
            for item in chunk:
                if self.contains(item):
                    count += 1
            return count

        chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
        with ThreadPool(processes=NUMBER_OF_THREADS) as pool:
            results = pool.map(process_chunk, chunks)
        return sum(results)


"""
####################################################################################################
    BLOOM FILTER SEQUENZIALE
####################################################################################################
"""


class BloomFilterSeq(BloomFilter):
    def add_from_file(self, items: list[str]) -> None:
        for item in items:
            self.add(item)

    def contains_from_file(self, items: list[str]) -> int:
        count = 0
        for item in items:
            if self.contains(item):
                count += 1
        return count


def main():
    if sys._is_gil_enabled():
        print("GIL enabled, exiting program...")
        sys.exit(-1)

    filename = "../rockyou.txt"
    ctrl_filename = "../parole_uniche.txt"

    passwords = []  # passwords da inserire nel dizionario
    ctrl_passwords = (
        []
    )  # passwords di controllo (ognuna composta da 8 caratteri alfabetici genereati casualmente)

    num_cycles = 2  # numero di cicli testing, allineato alla versione C++ per rendere le medie comparabili

    # inizializzazione delle variabili di raccolta dei dati finali
    tot_add_time_par = 0.0
    tot_add_time_seq = 0.0
    tot_srch_time_par = 0.0
    tot_srch_time_seq = 0.0
    tot_speed_up_add = 0.0
    tot_speed_up_srch = 0.0
    tot_eff_add = 0.0
    tot_eff_srch = 0.0
    num_threads = NUMBER_OF_THREADS

    print(f"Caricamento password da '{filename}'...")
    passwords = load_passwords(filename, 0)

    if not passwords:
        print("caricamento fallito, l'array è vuoto.")
        return 1

    print(f"Caricate {len(passwords)} password.\n")

    # filter_size calcolato dinamicamente in base al numero di password effettivamente caricate,
    # con rapporto m/n = 10 (vicino all'ottimo teorico per k=6, vedi discussione precedente).
    # NOTA: size qui resta un numero di BIT logici, il bit-packing dentro BloomFilter si occupa
    # di allocare solo size/8 byte reali
    filter_size = len(passwords) * 10
    print(
        f"Filter size calcolato (in bit): {filter_size} -> circa {(filter_size + 7) // 8 // (1024 * 1024)} MB allocati\n"
    )

    print(f"Caricamento password da '{ctrl_filename}'...")
    ctrl_passwords = load_passwords(ctrl_filename, 0)

    if not ctrl_passwords:
        print("caricamento fallito, l'array di controllo è vuoto.")
        return 1

    print(f"Caricate {len(ctrl_passwords)} password.\n")

    # bloom parallelo creato una sola volta fuori dal ciclo e riutilizzato: cosi' evitiamo di
    # pagare ad ogni iterazione l'allocazione di NUM_LOCK_STRIPES lock, in modo analogo a come
    # in OpenMP il team di thread e' tipicamente persistente tra le region parallele
    for i in range(num_cycles):
        print(f"\n=== INIZIO CICLO SPERIMENTALE {i + 1}/{num_cycles} ===", flush=True)
        # ======================== PARTE PARALLELA ========================

        # inizializzazione variabili per la raccolta dati
        time_add_par = 0.0
        start_add_par = 0.0
        start_srch_par = 0.0
        time_srch_par = 0.0

        start_add_seq = 0.0
        start_srch_seq = 0.0
        time_add_seq = 0.0
        time_srch_seq = 0.0

        speedup_add = 0.0
        speedup_srch = 0.0  # speedup = tempo_op_seq / tempo_op_par
        eff_add = 0.0
        eff_srch = 0.0

        bloom = BloomFilterPar(filter_size)

        # operazione ADD per popolare il dizionario (parallelo)
        start_add_par = time.perf_counter()
        bloom.add_from_file(passwords)
        time_add_par = time.perf_counter() - start_add_par

        # controllo della presenza di passwords all'interno del dizionario (parallelo)
        start_srch_par = time.perf_counter()
        res_par = bloom.contains_from_file(ctrl_passwords)
        time_srch_par = time.perf_counter() - start_srch_par

        # ======================== PARTE SEQUENZIALE ========================

        bloom_seq = BloomFilterSeq(filter_size)

        # operazione ADD per popolare il dizionario (sequenziale)
        start_add_seq = time.perf_counter()
        bloom_seq.add_from_file(passwords)
        time_add_seq = time.perf_counter() - start_add_seq

        # controllo della presenza di passwords all'interno del dizionario (sequenziale)
        start_srch_seq = time.perf_counter()
        res_seq = bloom_seq.contains_from_file(ctrl_passwords)
        time_srch_seq = time.perf_counter() - start_srch_seq

        # calcolo delle metriche per ciclo
        speedup_add = time_add_seq / time_add_par if time_add_par > 0 else 0.0
        speedup_srch = time_srch_seq / time_srch_par if time_srch_par > 0 else 0.0
        eff_add = speedup_add / num_threads
        eff_srch = speedup_srch / num_threads

        # stampa delle variabili calcolate
        print(f"CICLO SPERIMENTALE NUMERO {i + 1}\n")

        print(f"Numero Threads: {num_threads}")
        print("Numero Places: N/A")
        print("Policy Attiva: GIL disabilitato (GIL = 0)\n")

        print("=== INSERIMENTO (ADD) ===")
        print(f"Tempo Sequenziale: {time_add_seq:.6f} s")
        print(f"Tempo Parallelo:   {time_add_par:.6f} s")
        print(f"Speedup Inserimento: {speedup_add:.2f}x")
        print(f"Efficiency:   {eff_add * 100:.2f}%\n")

        print("=== RICERCA (CONTAINS) ===")
        print(f"Tempo Sequenziale: {time_srch_seq:.6f} s")
        print(f"Tempo Parallelo:   {time_srch_par:.6f} s")
        print(f"Speedup Ricerca:   {speedup_srch:.2f}x")
        print(f"Efficiency:   {eff_srch * 100:.2f}%\n")

        print(
            f"Verifica correttezza (elementi trovati Seq vs Par): {res_seq} / {res_par}\n"
        )

        tot_add_time_seq += time_add_seq
        tot_add_time_par += time_add_par

        tot_srch_time_seq += time_srch_seq
        tot_srch_time_par += time_srch_par

        tot_speed_up_add += speedup_add
        tot_speed_up_srch += speedup_srch

        tot_eff_add += eff_add
        tot_eff_srch += eff_srch

    # stampa dei risultati (medi) finali
    print("=== RISULTATI FINALI ADD (MEDIA) ===\n")
    print(f"Tempo Medio Sequenziale: {tot_add_time_seq / num_cycles:.6f}s")
    print(f"Tempo Medio Parallelo: {tot_add_time_par / num_cycles:.6f}s")
    print(f"Speedup Medio: {tot_speed_up_add / num_cycles:.2f}x")
    print(f"Effeciency Media: {(tot_eff_add / num_cycles) * 100:.2f}%\n")

    print("=== RISULTATI FINALI CONTAINS (MEDIA) ===\n")
    print(f"Tempo Medio Sequenziale: {tot_srch_time_seq / num_cycles:.6f}s")
    print(f"Tempo Medio Parallelo: {tot_srch_time_par / num_cycles:.6f}s")
    print(f"Speedup Medio: {tot_speed_up_srch / num_cycles:.2f}x")
    print(f"Effeciency Media: {(tot_eff_srch / num_cycles) * 100:.2f}%")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEsecuzione interrotta dall'utente.")
        sys.exit(0)
