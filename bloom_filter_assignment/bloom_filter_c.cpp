#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <omp.h>
#include <functional> // per std::hash
#include <ctime> // per std::clock
#include <cstdint> // per uint8_t
#include <array>   // per std::array
#include "library/MurmurHash3.h"


/*
###################################################
OMP_PROC_BIND="spread"
OMP_PLACES=threads
###################################################
*/

constexpr size_t K_HASHES = 6;  // numero di funzioni hash

// funzione di supporto per caricare le password da file .txt
std::vector<std::string> load_passwords(const std::string& filename, size_t max_lines = 0) {
    std::vector<std::string> passwords;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire il file '" << filename << "'!\n";
        return passwords;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            // Rimuove l'eventuale carattere '\r' da file formattati in windows
            if (line.back() == '\r') line.pop_back();
            passwords.push_back(line);
            if (max_lines > 0 && passwords.size() >= max_lines) break;
        }
    }
    return passwords;
}

/*####################################################################################################
    BLOOM FILTER ASTRATTO
####################################################################################################*/
class BloomFilter {
protected:
    size_t size; // dimensione del bit_array
    std::vector<uint8_t> bit_array;

    // Calcola l'hash a 128 bit una sola volta e restituisce la coppia {h1, h2}
    std::array<uint64_t, 2> _hash(const uint8_t* data, std::size_t len) const {
        std::array<uint64_t, 2> hashValue;
        MurmurHash3_x64_128(data, len, 0, hashValue.data());
        return hashValue;
    }

    // Combina h1 e h2 con l'indice (Double Hashing) restituendo solo l'indice calcolato
    size_t _hash_single(uint64_t h1, uint64_t h2, size_t hash_idx) const {
        return static_cast<size_t>(h1 + hash_idx * h2) % size;
    }

public:
    explicit BloomFilter(size_t size) : size(size), bit_array(size, 0) {}
    
    // distruttore
    virtual ~BloomFilter() = default;

    // metodi di default
    virtual void add(const std::string& item) {
        auto [h1, h2] = _hash(reinterpret_cast<const uint8_t*>(item.data()), item.size());
        for (size_t j = 0; j < K_HASHES; ++j) {
            bit_array[_hash_single(h1, h2, j)] = 1;
        }
    }

    virtual bool contains(const std::string& item) const {
        auto [h1, h2] = _hash(reinterpret_cast<const uint8_t*>(item.data()), item.size());
        for (size_t j = 0; j < K_HASHES; ++j) {
            if (bit_array[_hash_single(h1, h2, j)] == 0) {
                return false;
            }
        }
        return true;
    }

    // metodi virtuali
    virtual void add_from_file(const std::vector<std::string>& items) = 0;
    virtual size_t contains_from_file(const std::vector<std::string>& items) const = 0;
};

/*####################################################################################################
    BLOOM FILTER PARALLELO
####################################################################################################*/
class BloomFilterPar : public BloomFilter{ 
public:
    using BloomFilter::BloomFilter; 

    void add_from_file(const std::vector<std::string>& items) override {
        #pragma omp parallel num_threads(omp_get_max_threads())
        #pragma omp for schedule(dynamic, 1024)
        for (size_t i = 0; i < items.size(); ++i) {
            auto [h1, h2] = _hash(reinterpret_cast<const uint8_t*>(items[i].data()), items[i].size());
            for (size_t j = 0; j < K_HASHES; ++j) {
                size_t index = _hash_single(h1, h2, j);
                bit_array[index] = 1;
            }
        }
    }

    // ricerca di un batch di password nel Bloom Filter 
    size_t contains_from_file(const std::vector<std::string>& items) const {
        size_t count = 0;
        #pragma omp parallel num_threads(omp_get_max_threads())
        #pragma omp for schedule(dynamic, 1024) reduction(+ : count) // reduction su count per renderla affidabile senza utilizzare sincronizzazione
        for (size_t i = 0; i < items.size(); ++i) {
            if (contains(items[i])) {
                count++;
            }
        }
        return count;
    }
};

/*####################################################################################################
    BLOOM FILTER SEQUENZIALE
####################################################################################################*/

class BloomFilterSeq : public BloomFilter{ 
public:
    using BloomFilter::BloomFilter;

    void add_from_file(const std::vector<std::string>& items) override {
        for (const auto& item : items) {
            add(item);
        }
    }

    size_t contains_from_file(const std::vector<std::string>& items) const override {
        size_t count = 0;
        for (const auto& item : items) {
            if (contains(item)) {
                count++;
            }
        }
        return count;
    }
};

int main() {
    const std::string filename = "rockyou.txt";
    const std::string ctrl_filename = "parole_uniche.txt";

    size_t filter_size = 143443800; // più o meno 17MB

    std::vector<std::string> passwords; // passwords da inserire nel dizionario
    std::vector<std::string> ctrl_passwords; // passwords di controllo (ognuna composta da 8 caratteri alfabetici genereati casualmente)

    int num_cycles = 15; // numero di cicli testing

    // inizializzazione delle variabili di raccolta dei dati finali
    double tot_add_time_par = 0, tot_add_time_seq = 0;

    double tot_srch_time_par = 0, tot_srch_time_seq = 0;

    double tot_speed_up_add = 0, tot_speed_up_srch = 0;

    double tot_eff_add = 0, tot_eff_srch = 0;

    int num_threads = omp_get_max_threads();
 
    std::cout << "Caricamento password da '" << filename << "'...\n";
    passwords = load_passwords(filename, 0);

    if(passwords.empty())
    {
        std::cout << "caricamento fallito, l'array è vuoto.";
        return 1;
    }

    std::cout << "Caricate " << passwords.size() << " password.\n\n";

    std::cout << "Caricamento password da '" << ctrl_filename << "'...\n";
    ctrl_passwords = load_passwords(ctrl_filename, 0);

    if(ctrl_passwords.empty())
    {
        std::cout << "caricamento fallito, l'array di controllo è vuoto.";
        return 1;
    }

    std::cout << "Caricate " << ctrl_passwords.size() << " password.\n\n";

    for(int i = 0; i < num_cycles; i++){

        // ======================== PARTE PARALLELA ========================

        // inizializzazione variabili per la raccolta dati
        double time_add_par = 0, time_add_par_cpu = 0; 
        double start_add_par = 0, start_srch_par = 0;
        double time_srch_par_cpu = 0, time_srch_par = 0;

        double start_add_seq = 0, start_srch_seq = 0;
        double time_add_seq = 0, time_add_seq_cpu = 0;
        double time_srch_seq_cpu = 0, time_srch_seq = 0;

        double speedup_add = 0, speedup_srch = 0; // speedup = tempo_op_seq / tempo_op_par
        double eff_add = 0, eff_srch = 0;

        std::clock_t start_add_par_cpu, start_srch_par_cpu, start_add_seq_cpu, start_srch_seq_cpu;

        BloomFilterPar bloom(filter_size);

        // operazione ADD per popolare il dizionario (parallelo)
        start_add_par = omp_get_wtime();
        start_add_par_cpu = std::clock();

        bloom.add_from_file(passwords);

        time_add_par = omp_get_wtime() - start_add_par;
        time_add_par_cpu = double(std::clock() - start_add_par_cpu) / CLOCKS_PER_SEC;

        // controllo della presenza di passwords all'interno del dizionario (parallelo)
        start_srch_par = omp_get_wtime();
        start_srch_par_cpu = std::clock();

        size_t res_par = bloom.contains_from_file(ctrl_passwords);

        time_srch_par_cpu = double(std::clock() - start_srch_par_cpu) / CLOCKS_PER_SEC;
        time_srch_par = omp_get_wtime() - start_srch_par;

        // ======================== PARTE SEQUENZIALE ========================

        BloomFilterSeq bloom_seq(filter_size);

        // operazione ADD per popolare il dizionario (sequenziale)
        start_add_seq = omp_get_wtime();
        start_add_seq_cpu = std::clock();

        bloom_seq.add_from_file(passwords);

        time_add_seq = omp_get_wtime() - start_add_seq;
        time_add_seq_cpu = double(std::clock() - start_add_seq_cpu) / CLOCKS_PER_SEC;

        // controllo della presenza di passwords all'interno del dizionario (sequenziale)
        start_srch_seq = omp_get_wtime();
        start_srch_seq_cpu = std::clock();

        size_t res_seq = bloom_seq.contains_from_file(ctrl_passwords);

        time_srch_seq_cpu = double(std::clock() - start_srch_seq_cpu) / CLOCKS_PER_SEC;
        time_srch_seq = omp_get_wtime() - start_srch_seq;

        // calcolo delle metriche per ciclo
        speedup_add = time_add_seq / time_add_par;
        speedup_srch = time_srch_seq / time_srch_par;
        eff_add = speedup_add / num_threads;
        eff_srch = speedup_srch / num_threads;

        // stampa delle variabili calcolate
        std::cout << "CICLO SPERIMENTALE NUMERO " << i + 1 << "\n\n";

        std::cout << "Numero Threads: " << num_threads << "\n";
        std::cout << "Numero Places: " << omp_get_num_places() << "\n";
        std::cout << "Policy Attiva: " << omp_get_proc_bind() << "\n\n";

        std::cout << "=== INSERIMENTO (ADD) ===\n";
        std::cout << "Tempo Sequenziale: " << time_add_seq << " s\n";
        std::cout << "Tempo Parallelo:   " << time_add_par << " s\n";
        std::cout << "Tempo CPU Sequenziale: " << time_add_seq_cpu << " s\n";
        std::cout << "Tempo CPU Parallelo:   " << time_add_par_cpu << " s\n";
        std::cout << "Speedup Inserimento: " << speedup_add << "x\n";
        std::cout << "Efficiency:   " << eff_add * 100 << "%\n\n";

        std::cout << "=== RICERCA (CONTAINS) ===\n";
        std::cout << "Tempo Sequenziale: " << time_srch_seq << " s\n";
        std::cout << "Tempo Parallelo:   " << time_srch_par << " s\n";
        std::cout << "Tempo CPU Sequenziale: " << time_srch_seq_cpu << " s\n";
        std::cout << "Tempo CPU Parallelo:   " << time_srch_par_cpu << " s\n";
        std::cout << "Speedup Ricerca:   " << speedup_srch << "x\n";
        std::cout << "Efficiency:   " << eff_srch * 100 << "%\n\n";

        std::cout << "Verifica correttezza (elementi trovati Seq vs Par): " 
                    << res_seq << " / " << res_par << "\n\n";

        tot_add_time_seq += time_add_seq;
        tot_add_time_par += time_add_par;

        tot_srch_time_seq += time_srch_seq;
        tot_srch_time_par += time_srch_par;

        tot_speed_up_add += speedup_add;
        tot_speed_up_srch += speedup_srch;

        tot_eff_add += eff_add;
        tot_eff_srch += eff_srch;
    }

    // stampa dei risultati (medi) finali
    std::cout << "=== RISULTATI FINALI ADD (MEDIA) ===\n\n";
    std::cout << "Tempo Medio Sequenziale: " << tot_add_time_seq / num_cycles << "s \n";
    std::cout << "Tempo Medio Parallelo: " << tot_add_time_par / num_cycles << "s \n";
    std::cout << "Speedup Medio: " << tot_speed_up_add / num_cycles << "x \n";
    std::cout << "Effeciency Media: " << (tot_eff_add / num_cycles) * 100 << "\n";

    std::cout << "=== RISULTATI FINALI CONTAINS (MEDIA) ===\n\n";
    std::cout << "Tempo Medio Sequenziale: " << tot_srch_time_seq / num_cycles << "s \n";
    std::cout << "Tempo Medio Parallelo: " << tot_srch_time_par / num_cycles << "s \n";
    std::cout << "Speedup Medio: " << tot_speed_up_srch / num_cycles << "x \n";
    std::cout << "Effeciency Media: " << (tot_eff_srch / num_cycles) * 100 << "\n";
    
    return 0;
}