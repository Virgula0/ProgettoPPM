#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <omp.h>
#include <array>
#include <cstdint> // per uint8_t
#include <array>   // per array

#include "libraries/MurmurHash3.h"

using namespace std;

#define NUMBER_OF_THREADS omp_get_max_threads()

/*
###################################################
    OMP_PROC_BIND="spread"
    OMP_PLACES=threads
###################################################
*/

constexpr size_t K_HASHES = 7; // numero di funzioni hash

// funzione di supporto per caricare le password da file .txt
vector<string> load_passwords(const string& filename, size_t max_lines = 0) {
    vector<string> passwords;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Errore: impossibile aprire il file '" << filename << "'!\n";
        return passwords;
    }

    string line;
    while (getline(file, line)) {
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
    BLOOM FILTER ASTRATTO
####################################################################################################*/
class BloomFilter {
protected:
    size_t size; // dimensione del bit_array
    vector<uint8_t> bit_array;

    // Calcola l'hash a 128 bit una sola volta e restituisce la coppia {h1, h2}
    array<uint64_t, 2> _hash(const uint8_t* data, size_t len) const {
        array<uint64_t, 2> hashValue;
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
    virtual void add(const string& item) {
        auto [h1, h2] = _hash(reinterpret_cast<const uint8_t*>(item.data()), item.size());
        for (size_t j = 0; j < K_HASHES; ++j) {
            bit_array[_hash_single(h1, h2, j)] = 1;
        }
    }

    virtual bool contains(const string& item) const {
        auto [h1, h2] = _hash(reinterpret_cast<const uint8_t*>(item.data()), item.size());
        for (size_t j = 0; j < K_HASHES; ++j) {
            if (bit_array[_hash_single(h1, h2, j)] == 0) {
                return false;
            }
        }
        return true;
    }

    // metodi virtuali
    virtual void add_from_file(const vector<string>& items) = 0;
    virtual size_t contains_from_file(const vector<string>& items) const = 0;
};

/*####################################################################################################
    BLOOM FILTER PARALLELO
####################################################################################################*/
class BloomFilterPar : public BloomFilter {
public:
    using BloomFilter::BloomFilter;

    void add_from_file(const vector<string>& items) override {
#pragma omp parallel num_threads(NUMBER_OF_THREADS)
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
    size_t contains_from_file(const vector<string>& items) const override {
        size_t count = 0;
#pragma omp parallel num_threads(NUMBER_OF_THREADS)
#pragma omp for schedule(dynamic, 1024)                                                                                \
        reduction(+ : count) // reduction su count per renderla affidabile senza utilizzare sincronizzazione
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

class BloomFilterSeq : public BloomFilter {
public:
    using BloomFilter::BloomFilter;

    void add_from_file(const vector<string>& items) override {
        for (const auto& item : items) {
            add(item);
        }
    }

    size_t contains_from_file(const vector<string>& items) const override {
        size_t count = 0;
        for (const auto& item : items) {
            if (contains(item)) {
                count++;
            }
        }
        return count;
    }
};

size_t calculate_filter_size_dimention(const vector<string>& items) {
    // p_falso_positivo =  (1 - e^(-k*n/m))^k // n elementi inseriti, k iterazioni di hash, m dimensione del bit array
    // con m/n = 0.8 = 10 il p_falso_positivo e' 0.8, accettabile
    // k ottimale => k_opt = (m/n) * ln(2) = 10 * 0.693 ovvero circa 7
    return items.size() * 10;
}


int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Atteso in posizione 2 degli argomenti il dizionario da caricare" << endl
             << "Atteso in posizione 3 degli argomenti il dizionario di controllo da caricare" << endl;
        return -1;
    }

    const string filename = argv[1];      // rockyou.txt
    const string ctrl_filename = argv[2]; // parole_uniche.txt

    vector<string> passwords; // passwords da inserire nel dizionario
    vector<string>
            ctrl_passwords; // passwords di controllo (ognuna composta da 8 caratteri alfabetici genereati casualmente)

    int num_cycles = 15; // numero di cicli testing

    // inizializzazione delle variabili di raccolta dei dati finali
    double tot_add_time_par = 0, tot_add_time_seq = 0;
    double tot_srch_time_par = 0, tot_srch_time_seq = 0;
    double tot_speed_up_add = 0, tot_speed_up_srch = 0;
    double tot_eff_add = 0, tot_eff_srch = 0;
    int num_threads = NUMBER_OF_THREADS;

    cout << "Caricamento password da '" << filename << "'...\n";
    passwords = load_passwords(filename, 0);

    if (passwords.empty()) {
        cout << "caricamento fallito, l'array è vuoto.";
        return 1;
    }

    size_t filter_size = calculate_filter_size_dimention(passwords); // più o meno 17MB per rockyou

    cout << "Filter size calcolato " << filter_size << endl;
    cout << "Caricate " << passwords.size() << " password.\n\n";

    cout << "Caricamento ctrl_passwords da '" << ctrl_filename << "'...\n";
    ctrl_passwords = load_passwords(ctrl_filename, 0);

    if (ctrl_passwords.empty()) {
        cout << "caricamento fallito, l'array di controllo è vuoto.";
        return 1;
    }

    cout << "Caricate " << ctrl_passwords.size() << " password.\n\n";

    for (int i = 0; i < num_cycles; i++) {
        // ======================== PARTE PARALLELA ========================

        // inizializzazione variabili per la raccolta dati
        double time_add_par = 0;
        double start_add_par = 0, start_srch_par = 0;
        double time_srch_par = 0;

        double start_add_seq = 0, start_srch_seq = 0;
        double time_add_seq = 0;
        double time_srch_seq = 0;

        double speedup_add = 0, speedup_srch = 0; // speedup = tempo_op_seq / tempo_op_par
        double eff_add = 0, eff_srch = 0;

        BloomFilterPar bloom(filter_size);

        // operazione ADD per popolare il dizionario (parallelo)
        start_add_par = omp_get_wtime();
        bloom.add_from_file(passwords);
        time_add_par = omp_get_wtime() - start_add_par;

        // controllo della presenza di passwords all'interno del dizionario (parallelo)
        start_srch_par = omp_get_wtime();
        size_t res_par = bloom.contains_from_file(ctrl_passwords);
        time_srch_par = omp_get_wtime() - start_srch_par;

        // ======================== PARTE SEQUENZIALE ========================

        BloomFilterSeq bloom_seq(filter_size);

        // operazione ADD per popolare il dizionario (sequenziale)
        start_add_seq = omp_get_wtime();
        bloom_seq.add_from_file(passwords);
        time_add_seq = omp_get_wtime() - start_add_seq;

        // controllo della presenza di passwords all'interno del dizionario (sequenziale)
        start_srch_seq = omp_get_wtime();
        size_t res_seq = bloom_seq.contains_from_file(ctrl_passwords);
        time_srch_seq = omp_get_wtime() - start_srch_seq;

        // calcolo delle metriche per ciclo
        speedup_add = time_add_seq / time_add_par;
        speedup_srch = time_srch_seq / time_srch_par;
        eff_add = speedup_add / num_threads;
        eff_srch = speedup_srch / num_threads;

        // stampa delle variabili calcolate
        cout << "CICLO SPERIMENTALE NUMERO " << i + 1 << "\n\n";

        cout << "Numero Threads: " << num_threads << "\n";
        cout << "Numero Places: " << omp_get_num_places() << "\n";
        cout << "Policy Attiva: " << omp_get_proc_bind() << "\n\n";

        cout << "=== INSERIMENTO (ADD) ===\n";
        cout << "Tempo Sequenziale: " << time_add_seq << " s\n";
        cout << "Tempo Parallelo:   " << time_add_par << " s\n";
        cout << "Speedup Inserimento: " << speedup_add << "x\n";
        cout << "Efficiency:   " << eff_add * 100 << "%\n\n";

        cout << "=== RICERCA (CONTAINS) ===\n";
        cout << "Tempo Sequenziale: " << time_srch_seq << " s\n";
        cout << "Tempo Parallelo:   " << time_srch_par << " s\n";
        cout << "Speedup Ricerca:   " << speedup_srch << "x\n";
        cout << "Efficiency:   " << eff_srch * 100 << "%\n\n";

        cout << "Verifica correttezza (elementi trovati Seq vs Par): " << res_seq << " / " << res_par << "\n\n";

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
    cout << "=== RISULTATI FINALI ADD (MEDIA) ===\n\n";
    cout << "Tempo Medio Sequenziale: " << tot_add_time_seq / num_cycles << "s \n";
    cout << "Tempo Medio Parallelo: " << tot_add_time_par / num_cycles << "s \n";
    cout << "Speedup Medio: " << tot_speed_up_add / num_cycles << "x \n";
    cout << "Effeciency Media: " << (tot_eff_add / num_cycles) * 100 << "\n";

    cout << "=== RISULTATI FINALI CONTAINS (MEDIA) ===\n\n";
    cout << "Tempo Medio Sequenziale: " << tot_srch_time_seq / num_cycles << "s \n";
    cout << "Tempo Medio Parallelo: " << tot_srch_time_par / num_cycles << "s \n";
    cout << "Speedup Medio: " << tot_speed_up_srch / num_cycles << "x \n";
    cout << "Effeciency Media: " << (tot_eff_srch / num_cycles) * 100 << "\n";

    return 0;
}