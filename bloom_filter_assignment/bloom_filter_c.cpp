#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <omp.h>
#include <functional> // Per std::hash
#include <ctime>
#include <cstdint> // Necessario per uint8_t

/*
###################################################
OMP_PROC_BIND="spread"
OMP_PLACES - non modificato
###################################################
*/

constexpr size_t K_HASHES = 6; // numero di funzioni hash

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
            if (line.back() == '\r')
                line.pop_back();
            passwords.push_back(line);
            if (max_lines > 0 && passwords.size() >= max_lines)
                break;
        }
    }
    return passwords;
}
/*####################################################################################################
    BLOOM FILTER PARALLELO
####################################################################################################*/
class BloomFilterPar {
private:
    size_t size; // custom size del bit array
    std::vector<uint8_t> bit_array;

    // restituisce k indici univoci per un dato elemento
    size_t _hash_single(const std::string& item, size_t hash_idx) const {
        size_t h1 = std::hash<std::string>{}(item);
        size_t h2 = std::hash<size_t>{}(h1 ^ 0x9e3779b97f4a7c15ULL);
        return (h1 + hash_idx * h2) % size;
    }

public:
    // costruttore che inizializza il vettore alla dimensione desiderata con tutti bit a false
    explicit BloomFilterPar(size_t size) : size(size), bit_array(size, 0) {}

    void add_from_file(const std::vector<std::string>& items) {
#pragma omp parallel num_threads(omp_get_max_threads())
#pragma omp for schedule(dynamic, 1024) collapse(2)
        for (size_t i = 0; i < items.size(); ++i) {
            for (size_t j = 0; j < K_HASHES; ++j) {
                size_t index = _hash_single(items[i], j);
                bit_array[index] = 1;
            }
        }
    }

    // ricerca di un batch di password nel Bloom Filter
    size_t contains_from_file(const std::vector<std::string>& items) const {
        size_t count = 0;
#pragma omp parallel num_threads(omp_get_max_threads())
#pragma omp for schedule(dynamic, 1024) reduction(+ : count)
        for (size_t i = 0; i < items.size(); ++i) {
            if (contains(items[i])) {
                count++;
            }
        }
        return count;
    }

    // verifica la presenza della password analizzando i suoi 3 bit associati
    bool contains(const std::string& item) const {
        for (size_t j = 0; j < K_HASHES; ++j) {
            size_t index = _hash_single(item, j);
            if (bit_array[index] == 0) {
                return false;
            }
        }
        return true;
    }
};

/*####################################################################################################
    BLOOM FILTER SEQUENZIALE
####################################################################################################*/

class BloomFilterSeq {
private:
    size_t size; // custom size del bit array
    std::vector<uint8_t> bit_array;

    // genera k indici univoci per un dato elemento
    size_t _hash_single(const std::string& item, size_t hash_idx) const {
        size_t h1 = std::hash<std::string>{}(item);
        size_t h2 = std::hash<size_t>{}(h1 ^ 0x9e3779b97f4a7c15ULL);
        return (h1 + hash_idx * h2) % size;
    }

public:
    // costruttore che inizializza il vettore alla dimensione indicata con tutti bit a false
    explicit BloomFilterSeq(size_t size) : size(size), bit_array(size, 0) {}

    void add(const std::string& item) {
        for (size_t j = 0; j < K_HASHES; ++j) {
            size_t index = _hash_single(item, j);
            bit_array[index] = 1;
        }
    }

    bool contains(const std::string& item) const {
        for (size_t j = 0; j < K_HASHES; ++j) {
            size_t index = _hash_single(item, j);
            if (bit_array[index] == 0) {
                return false;
            }
        }
        return true;
    }
};

int main() {
    const std::string filename = "rockyou.txt";
    const std::string ctrl_filename = "1_million_passwords.txt";

    size_t filter_size = 143443800; // più o meno 17MB

    std::vector<std::string> passwords;
    std::vector<std::string> ctrl_passwords;

    int num_cycles = 5;

    double tot_add_time_par = 0, tot_add_time_seq = 0;

    double tot_srch_time_par = 0, tot_srch_time_seq = 0;

    double tot_speed_up_add = 0, tot_speed_up_srch = 0;

    double tot_eff_add = 0, tot_eff_srch = 0;

    int num_threads = omp_get_max_threads();

    std::cout << "Caricamento password da '" << filename << "'...\n";
    passwords = load_passwords(filename, 0);

    if (passwords.empty()) {
        std::cout << "caricamento fallito, l'array è vuoto.";
        return 1;
    }

    std::cout << "Caricate " << passwords.size() << " password.\n\n";

    std::cout << "Caricamento password da '" << ctrl_filename << "'...\n";
    ctrl_passwords = load_passwords(ctrl_filename, 0);

    if (ctrl_passwords.empty()) {
        std::cout << "caricamento fallito, l'array di controllo è vuoto.";
        return 1;
    }

    std::cout << "Caricate " << ctrl_passwords.size() << " password.\n\n";

    for (int i = 0; i < num_cycles; i++) {
        double time_add_par = 0, time_add_par_cpu = 0;
        double start_add_par = 0, start_srch_par = 0;
        double time_srch_par_cpu = 0, time_srch_par = 0;

        double start_add_seq = 0, start_srch_seq = 0;
        double time_add_seq = 0, time_add_seq_cpu = 0;
        double time_srch_seq_cpu = 0, time_srch_seq = 0;

        double speedup_add = 0, speedup_srch = 0;
        double eff_add = 0, eff_srch = 0;

        std::clock_t start_add_par_cpu, start_srch_par_cpu, start_add_seq_cpu, start_srch_seq_cpu;

        BloomFilterPar bloom(filter_size);

        // Inserimento elementi
        start_add_par = omp_get_wtime();
        start_add_par_cpu = std::clock();

        bloom.add_from_file(passwords);

        time_add_par = omp_get_wtime() - start_add_par;
        time_add_par_cpu = double(std::clock() - start_add_par_cpu) / CLOCKS_PER_SEC;

        // Test di verifica
        start_srch_par = omp_get_wtime();
        start_srch_par_cpu = std::clock();

        size_t res_par = bloom.contains_from_file(ctrl_passwords);

        time_srch_par_cpu = double(std::clock() - start_srch_par_cpu) / CLOCKS_PER_SEC;
        time_srch_par = omp_get_wtime() - start_srch_par;

        BloomFilterSeq bloom_seq(filter_size);

        start_add_seq = omp_get_wtime();
        start_add_seq_cpu = std::clock();

        for (const auto& pwd : passwords) {
            bloom_seq.add(pwd);
        }

        time_add_seq = omp_get_wtime() - start_add_seq;
        time_add_seq_cpu = double(std::clock() - start_add_seq_cpu) / CLOCKS_PER_SEC;

        start_srch_seq = omp_get_wtime();
        start_srch_seq_cpu = std::clock();

        size_t found_seq = 0;
        for (const auto& pwd : ctrl_passwords) {
            if (bloom_seq.contains(pwd))
                found_seq++;
        }

        time_srch_seq_cpu = double(std::clock() - start_srch_seq_cpu) / CLOCKS_PER_SEC;
        time_srch_seq = omp_get_wtime() - start_srch_seq;

        speedup_add = time_add_seq / time_add_par;
        speedup_srch = time_srch_seq / time_srch_par;
        eff_add = speedup_add / num_threads;
        eff_srch = speedup_srch / num_threads;

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

        std::cout << "Verifica correttezza (elementi trovati Seq vs Par): " << found_seq << " / " << res_par << "\n\n";

        tot_add_time_seq += time_add_seq;
        tot_add_time_par += time_add_par;

        tot_srch_time_seq += time_srch_seq;
        tot_srch_time_par += time_srch_par;

        tot_speed_up_add += speedup_add;
        tot_speed_up_srch += speedup_srch;

        tot_eff_add += eff_add;
        tot_eff_srch += eff_srch;
    }

    // ADD – statistiche finali basate sui totali
    double avg_seq_add = tot_add_time_seq / num_cycles;
    double avg_par_add = tot_add_time_par / num_cycles;
    double speedup_add_final = tot_add_time_seq / tot_add_time_par; // speedup globale
    double eff_add_final = speedup_add_final / num_threads;

    std::cout << "=== RISULTATI FINALI ADD ===\n\n";
    std::cout << "Tempo Medio Sequenziale: " << avg_seq_add << " s\n";
    std::cout << "Tempo Medio Parallelo:   " << avg_par_add << " s\n";
    std::cout << "Speedup (totale):        " << speedup_add_final << "x\n";
    std::cout << "Efficiency:              " << eff_add_final * 100 << "%\n\n";

    // CONTAINS – statistiche finali basate sui totali
    double avg_seq_srch = tot_srch_time_seq / num_cycles;
    double avg_par_srch = tot_srch_time_par / num_cycles;
    double speedup_srch_final = tot_srch_time_seq / tot_srch_time_par;
    double eff_srch_final = speedup_srch_final / num_threads;

    std::cout << "=== RISULTATI FINALI CONTAINS ===\n\n";
    std::cout << "Tempo Medio Sequenziale: " << avg_seq_srch << " s\n";
    std::cout << "Tempo Medio Parallelo:   " << avg_par_srch << " s\n";
    std::cout << "Speedup (totale):        " << speedup_srch_final << "x\n";
    std::cout << "Efficiency:              " << eff_srch_final * 100 << "%\n";

    return 0;
}