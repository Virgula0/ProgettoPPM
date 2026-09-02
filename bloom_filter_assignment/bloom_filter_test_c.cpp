#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <omp.h>
#include <functional> // Per std::hash
#include <ctime>

/*
###################################################
OMP_PROC_BIND="spread"
OMP_PLACES - non modificato
###################################################
*/

// Funzione di supporto per caricare le password dal file rockyou.txt
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
            // Rimuove l'eventuale carattere '\r' da file formattati in Windows
            if (line.back() == '\r') line.pop_back();
            passwords.push_back(line);
            if (max_lines > 0 && passwords.size() >= max_lines) break;
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
    std::vector<bool> bit_array; 
    const size_t k = 6;          // numero di funzioni hash

    // restituisce k indici univoci per un dato elemento
    std::vector<size_t> _hashes(const std::string& item) const {
        std::vector<size_t> indices;
        indices.reserve(k);
        std::string salted_input;
        size_t hash_val;

    
        for (size_t i = 0; i < k; ++i) {
            salted_input = std::to_string(i) + ":" + item; // crea un input unico per ogni iterazione

            hash_val = std::hash<std::string>{}(salted_input);

            size_t index = hash_val % size; // modulo per rientrare nel range dell'array [0, size - 1]
            indices.push_back(index);
        }
    
        return indices;
    }

public:
    // costruttore che inizializza il vettore alla dimensione desiderata con tutti bit a false
    explicit BloomFilterPar(size_t size) 
        : size(size), bit_array(size, false) {}

    void add_from_file(const std::vector<std::string>& items) {
        #pragma omp parallel
        #pragma omp for schedule(static) 
        for(size_t i = 0; i < items.size(); ++i){
            for (size_t index : _hashes(items[i])) { // imposta a True tutti i bit ritornati dalla funzione _hashes
                bit_array[index] = true;
            }
        }
    }

    // ricerca di un batch di password nel Bloom Filter 
    size_t contains_from_file(const std::vector<std::string>& items) const {
        size_t count = 0;
        #pragma omp parallel
        #pragma omp for schedule(dynamic) reduction(+ : count)
        for (size_t i = 0; i < items.size(); ++i) {
            if (contains(items[i])) {
                count++;
            }
        }
        return count;
    }

    // verifica la presenza della password analizzando i suoi 3 bit associati
    bool contains(const std::string& item) const {
        bool flg = true; // flag per indicare se una password è presente o meno
        //#pragma omp parallel for schedule(static)
        for (size_t index : _hashes(item)) {
            // Se anche solo un bit è false, l'elemento NON è mai stato inserito
            if (!bit_array[index]) {
                /*#pragma omp critical
                {
                    flg = false;
                }
                #pragma omp cancel for // se anche un solo bit è True non c'è bisogno di confrontare gli altri, si interrompe il ciclo*/
                flg = false;
                break;
            }
        }
        //#pragma cancellation point for // punto di ritrovo per gli altri threads che analizzano una password (che potrebbe essere) presente
        return flg;
    }
};

/*####################################################################################################
    BLOOM FILTER SEQUENZIALE
####################################################################################################*/

class BloomFilterSeq { 
private:
    size_t size; // custom size del bit array
    std::vector<bool> bit_array;
    const size_t k = 6;  // numero di funzioni hash

    // genera k indici univoci per un dato elemento
    std::vector<size_t> _hashes(const std::string& item) const {
        std::vector<size_t> indices;
        indices.reserve(k);

        for (size_t i = 0; i < k; ++i) {
            std::string salted_input = std::to_string(i) + ":" + item; // crea un input unico per ciascuna iterazione

            size_t hash_val = std::hash<std::string>{}(salted_input);

            size_t index = hash_val % size; // Modulo per rientrare nel range dell'array [0, size - 1]
            indices.push_back(index);
            
        }
        return indices;
    }

public:
    // costruttore che inizializza il vettore alla dimensione indicata con tutti bit a false
    explicit BloomFilterSeq(size_t size) 
        : size(size), bit_array(size, false) {}

    void add(const std::string& item) {
        for (size_t index : _hashes(item)) { // imposta a True i bit agli indici ritornati dalla funzione _hashes
            bit_array[index] = true;
        }
    }

    bool contains(const std::string& item) const {
        for (size_t index : _hashes(item)) {
            if (!bit_array[index]) {
                return false;
            }
        }
        return true;
    }
};

int main() {
    const std::string filename = "rockyou.txt";
    const std::string ctrl_filename = "1_million_passwords.txt";
    size_t filter_size = 143443800;
    std::vector<std::string> passwords;
    std::vector<std::string> ctrl_passwords;

    double tot_add_time_par = 0;
    double tot_add_time_seq = 0;

    double tot_srch_time_par = 0;
    double tot_srch_time_seq = 0;

    double tot_speed_up_add = 0;
    double tot_cycles = 0;
    double tot_speed_up_srch = 0;

    double tot_eff_add = 0;
    double tot_eff_srch = 0;
 
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

    for(int i = 0; i < 5; i++){

        BloomFilterPar bloom(filter_size);

        // Inserimento elementi
        double start_add_par = omp_get_wtime();
        std::clock_t start_add_par_cpu = std::clock();

        bloom.add_from_file(passwords);

        int num_threads = omp_get_max_threads();

        double time_add_par = omp_get_wtime() - start_add_par;
        double time_add_par_cpu = double(std::clock() - start_add_par_cpu) / CLOCKS_PER_SEC;

        // Test di verifica
        double start_con_par = omp_get_wtime();
        std::clock_t start_srch_par_cpu = std::clock();

        size_t res_par = bloom.contains_from_file(ctrl_passwords);

        double time_srch_par_cpu = double(std::clock() - start_srch_par_cpu) / CLOCKS_PER_SEC;
        double time_con_par = omp_get_wtime() - start_con_par;

        BloomFilterSeq bloom_seq(filter_size);

        double start_seq_add = omp_get_wtime();
        std::clock_t start_add_seq_cpu = std::clock();
        for (const auto& pwd : passwords) {
            bloom_seq.add(pwd);
        }

        double time_seq_add = omp_get_wtime() - start_seq_add;
        double time_add_seq_cpu = double(std::clock() - start_add_seq_cpu) / CLOCKS_PER_SEC;

        double start_seq_query = omp_get_wtime();
        std::clock_t start_srch_seq_cpu = std::clock();

        size_t found_seq = 0;
        for (const auto& pwd : ctrl_passwords) {
            if (bloom_seq.contains(pwd)) found_seq++;
        }

        double time_srch_seq_cpu = double(std::clock() - start_srch_seq_cpu) / CLOCKS_PER_SEC;
        double time_seq_query = omp_get_wtime() - start_seq_query;

        double speedup_add = time_seq_add / time_add_par;
        double speedup_srch = time_seq_query / time_con_par;
        double eff_add = speedup_add / num_threads;
        double eff_srch = speedup_srch / num_threads;

        std::cout << "CICLO SPERIMENTALE NUMERO " << i + 1 << "\n\n";

        std::cout << "Numero Threads: " << num_threads << "\n";
        std::cout << "Numero Places: " << omp_get_num_places() << "\n";
        std::cout << "Policy Attiva: " << omp_get_proc_bind() << "\n\n";

        std::cout << "=== INSERIMENTO (ADD) ===\n";
        std::cout << "Tempo Sequenziale: " << time_seq_add << " s\n";
        std::cout << "Tempo Parallelo:   " << time_add_par << " s\n";
        std::cout << "Tempo CPU Sequenziale: " << time_add_seq_cpu << " s\n";
        std::cout << "Tempo CPU Parallelo:   " << time_add_par_cpu << " s\n";
        std::cout << "Speedup Inserimento: " << speedup_add << "x\n";
        std::cout << "Efficiency:   " << eff_add * 100 << "%\n\n";

        std::cout << "=== RICERCA (CONTAINS) ===\n";
        std::cout << "Tempo Sequenziale: " << time_seq_query << " s\n";
        std::cout << "Tempo Parallelo:   " << time_con_par << " s\n";
        std::cout << "Tempo CPU Sequenziale: " << time_srch_seq_cpu << " s\n";
        std::cout << "Tempo CPU Parallelo:   " << time_srch_par_cpu << " s\n";
        std::cout << "Speedup Ricerca:   " << speedup_srch << "x\n";
        std::cout << "Efficiency:   " << eff_srch * 100 << "%\n\n";

        std::cout << "Verifica correttezza (elementi trovati Seq vs Par): " 
                    << found_seq << " / " << res_par << "\n\n";

        tot_add_time_seq += time_seq_add;
        tot_add_time_par += time_add_par;

        tot_srch_time_seq += time_seq_query;
        tot_srch_time_par += time_con_par;

        tot_speed_up_add += speedup_add;
        tot_speed_up_srch += speedup_srch;

        tot_eff_add += eff_add;
        tot_eff_srch += eff_srch;
        tot_cycles += 1;
    }

    std::cout << "=== RISULTATI FINALI ADD (MEDIA) ===\n\n";
    std::cout << "Tempo Medio Sequenziale: " << tot_add_time_seq / tot_cycles << "s \n";
    std::cout << "Tempo Medio Parallelo: " << tot_add_time_par / tot_cycles << "s \n";
    std::cout << "Speedup Medio: " << tot_speed_up_add / tot_cycles << "x \n";
    std::cout << "Effeciency Media: " << (tot_eff_add / tot_cycles) * 100 << "\n";

    std::cout << "=== RISULTATI FINALI CONTAINS (MEDIA) ===\n\n";
    std::cout << "Tempo Medio Sequenziale: " << tot_srch_time_seq / tot_cycles << "s \n";
    std::cout << "Tempo Medio Parallelo: " << tot_srch_time_par / tot_cycles << "s \n";
    std::cout << "Speedup Medio: " << tot_speed_up_srch / tot_cycles << "x \n";
    std::cout << "Effeciency Media: " << (tot_eff_srch / tot_cycles) * 100 << "\n";
    
    return 0;
}