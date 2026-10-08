#ifndef _PROBLEM_DYNAMICWINDOWGCI_LOOKAHEAD_H
#define _PROBLEM_DYNAMICWINDOWGCI_LOOKAHEAD_H

// ============================================================================
// DECODER: DYNAMIC WINDOW GLOBAL CHEAPEST INSERTION WITH LOOKAHEAD
//
// Chromosome: size N + 2
// Genes 0..N-1: Product Priorities (Topological Urgency)
// Gene N: Window Factor Alpha (0.0 <= Alpha <= 1.0)
// Gene N+1: Lookahead Beta (0.0 <= Beta <= 1.0)
//
// Logic:
// 1. Same as DynamicWindowGCI but with Beta penalizing bad trailing setups.
//    FO = delta_cost + (beta * avg_future_cost)
// ============================================================================

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>

struct TProblemData
{
    int n; 

    int num_products;
    int num_machines;

    std::vector<double> machine_capacities; 
    std::vector<int> initial_state;         
    std::vector<double> demands;            
    
    std::vector<double> production_rates; 
    std::vector<double> setup_costs;
    std::vector<double> setup_times;
    
    double start_time = 0.0;
    double max_time = 1e9;  
};

void ReadData(char name[], TProblemData &data)
{
    std::ifstream file(name);
    if (!file.is_open()) {
        printf("\nERROR: File (%s) not found!\n", name);
        exit(1);
    }

    file >> data.num_products >> data.num_machines;
    
    // IMPORTANT: Chromosome size is N + 2 (N priorities + 1 alpha + 1 beta)
    data.n = data.num_products + 2;

    data.machine_capacities.resize(data.num_machines);
    for(int i=0; i<data.num_machines; ++i) file >> data.machine_capacities[i];

    data.initial_state.resize(data.num_machines);
    for(int i=0; i<data.num_machines; ++i) file >> data.initial_state[i];

    data.demands.resize(data.num_products);
    for(int i=0; i<data.num_products; ++i) file >> data.demands[i];

    data.production_rates.resize(data.num_products * data.num_machines);
    for(int i=0; i<data.num_products; ++i) {
        for(int m=0; m<data.num_machines; ++m) {
            file >> data.production_rates[i * data.num_machines + m];
        }
    }

    data.setup_costs.resize(data.num_products * data.num_products * data.num_machines);
    std::string label;
    for(int m=0; m<data.num_machines; ++m) {
        file >> label;
        for(int i=0; i<data.num_products; ++i) {
            for(int j=0; j<data.num_products; ++j) {
                int idx = (i * data.num_products * data.num_machines) + (j * data.num_machines) + m;
                file >> data.setup_costs[idx];
            }
        }
    }

    data.setup_times.resize(data.num_products * data.num_products * data.num_machines);
    for(int m=0; m<data.num_machines; ++m) {
        file >> label;
        for(int i=0; i<data.num_products; ++i) {
            for(int j=0; j<data.num_products; ++j) {
                int idx = (i * data.num_products * data.num_machines) + (j * data.num_machines) + m;
                file >> data.setup_times[idx];
            }
        }
    }
    
    file.close();
}

double Decoder(TSol &s, const TProblemData &data)
{
    const int np = data.num_products;
    const int nm = data.num_machines;
    
    double alpha = s.rk[np]; 
    double beta = s.rk[np + 1] * 10.0; // Extrai beta e multiplica pelo fator
    
    std::vector<int> sorted_products(np);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) {
        return s.rk[i] < s.rk[j]; 
    });

    std::vector<bool> is_allocated(np, false);
    int allocated_count = 0;

    std::vector<std::vector<int>> machine_seqs(nm);
    std::vector<double> machine_loads(nm, 0.0);
    double total_setup_cost = 0.0;
    double penalty = 0.0;

    auto idx = [np, nm](int prev, int curr, int m) {
        return (prev * np * nm) + (curr * nm) + m;
    };

    while (allocated_count < np) {
        bool panic_mode = false;
        if (omp_get_wtime() - data.start_time >= data.max_time) {
            panic_mode = true;
        }

        int remaining = np - allocated_count;
        int k = std::max(3, (int)std::ceil(alpha * remaining)); // Força pelo menos 3 candidatos para que a escolha lookahead funcione
        if (remaining < 3) k = remaining; // Ajuste final
        if (panic_mode) k = 1;

        std::vector<int> candidates;
        candidates.reserve(k);
        for (int p : sorted_products) {
            if (!is_allocated[p]) {
                candidates.push_back(p);
                if ((int)candidates.size() == k) break;
            }
        }

        int best_p = -1;
        int best_m = -1;
        int best_pos = -1;
        double best_increase_FO = std::numeric_limits<double>::infinity();
        double best_increase_cost = 0.0;
        double best_increase_time = 0.0;

        for (int p : candidates) {
            double local_best_FO = std::numeric_limits<double>::infinity();
            double local_best_cost = std::numeric_limits<double>::infinity();
            int local_best_m = -1;
            int local_best_pos = -1;
            double local_best_time = 0.0;

            for (int m = 0; m < nm; ++m) {
                double rate = data.production_rates[p * nm + m];
                if (rate <= 1e-6) continue;

                double prod_time = data.demands[p] / rate;
                const auto& seq = machine_seqs[m];
                int seq_len = seq.size();

                for (int pos = 0; pos <= seq_len; ++pos) {
                    int prev = (pos == 0) ? data.initial_state[m] : seq[pos - 1];
                    int next = (pos < seq_len) ? seq[pos] : -1;

                    // Cost Delta (custo real imediato)
                    double cost_add = data.setup_costs[idx(prev, p, m)];
                    double cost_rem = 0.0;
                    if (next != -1) {
                        cost_add += data.setup_costs[idx(p, next, m)];
                        cost_rem = data.setup_costs[idx(prev, next, m)];
                    }
                    double delta_cost = cost_add - cost_rem;

                    // Lookahead: Average Future Cost
                    double avg_future_cost = 0.0;
                    if (pos == seq_len && remaining > 1) { // Só faz sentido no final e se sobrarem outras peças
                        double sum_future = 0.0;
                        int count_future = 0;
                        for (int q = 0; q < np; ++q) {
                            if (!is_allocated[q] && q != p) {
                                sum_future += data.setup_costs[idx(p, q, m)];
                                count_future++;
                            }
                        }
                        if (count_future > 0) {
                            avg_future_cost = sum_future / count_future;
                        }
                    }

                    // FO Calculation: Custo Imediato + Penalidade de futuro ruim
                    double FO = delta_cost + (beta * avg_future_cost);

                    // Time Delta
                    double time_add = data.setup_times[idx(prev, p, m)];
                    double time_rem = 0.0;
                    if (next != -1) {
                        time_add += data.setup_times[idx(p, next, m)];
                        time_rem = data.setup_times[idx(prev, next, m)];
                    }
                    double delta_time = time_add + prod_time - time_rem;

                    // Avaliação
                    if (machine_loads[m] + delta_time <= data.machine_capacities[m]) {
                        if (FO < local_best_FO) {
                            local_best_FO = FO;
                            local_best_cost = delta_cost; // Precisamos guardar o custo real!
                            local_best_m = m;
                            local_best_pos = pos;
                            local_best_time = delta_time;
                        }
                    }
                }
            }

            // Comparando a inserção global atual usando a nova FO
            if (local_best_FO < best_increase_FO) {
                best_increase_FO = local_best_FO;
                best_increase_cost = local_best_cost; // Storing actual cost
                best_increase_time = local_best_time;
                best_p = p;
                best_m = local_best_m;
                best_pos = local_best_pos;
            }
        } 

        // Efetuando a locação global
        if (best_p != -1) {
            machine_seqs[best_m].insert(machine_seqs[best_m].begin() + best_pos, best_p);
            machine_loads[best_m] += best_increase_time;
            total_setup_cost += best_increase_cost; // OFV é baseada somente no custo real
            is_allocated[best_p] = true;
            allocated_count++;
        } else {
            // Panic force allocate
            int panic_p = candidates[0];
            penalty += 1e9 + data.demands[panic_p] * 1000.0;
            is_allocated[panic_p] = true;
            allocated_count++;
        }
    }

    return total_setup_cost + penalty;
}

void FreeMemoryProblem(TProblemData &data)
{
    data.machine_capacities.clear();
    data.initial_state.clear();
    data.demands.clear();
    data.production_rates.clear();
    data.setup_costs.clear();
    data.setup_times.clear();
}

void PrintSolution(TSol &s, const TProblemData &data)
{
    const int np = data.num_products;
    const int nm = data.num_machines;
    
    double alpha = s.rk[np];
    double beta = s.rk[np + 1] * 10.0; // Same beta
    
    printf("\n[CONFIRMAÇÃO] Iniciando reconstrução da solução com o Decoder Dinâmico Lookahead...\n");
    printf("[DynamicWindowGCI_Lookahead Alpha=%.4f Fator Beta (x10.0)=%.4f]\n", alpha, beta);
    
    double total_FO_tracked = 0.0; // Rastreando a FO total (Custo imediato + Penalidade)
    
    std::vector<int> sorted_products(np);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) {
        return s.rk[i] < s.rk[j];
    });

    std::vector<bool> is_allocated(np, false);
    int allocated_count = 0;
    std::vector<std::vector<int>> machine_seqs(nm);
    std::vector<double> machine_loads(nm, 0.0);

    auto idx = [np, nm](int prev, int curr, int m) {
        return (prev * np * nm) + (curr * nm) + m;
    };

    while (allocated_count < np) {
        bool panic_mode = false;
        int remaining = np - allocated_count;
        int k = std::max(3, (int)std::ceil(alpha * remaining)); // Força pelo menos 3 candidatos
        if (remaining < 3) k = remaining; // Ajuste final
        if (panic_mode) k = 1;

        std::vector<int> candidates;
        candidates.reserve(k);
        for (int p : sorted_products) {
            if (!is_allocated[p]) {
                candidates.push_back(p);
                if ((int)candidates.size() == k) break;
            }
        }

        int best_p = -1, best_m = -1, best_pos = -1;
        double best_FO = std::numeric_limits<double>::infinity();
        double best_cost = 0.0; // Guardando para separar FO vs Custo real
        double best_time = 0.0;

        for (int p : candidates) {
            for (int m = 0; m < nm; ++m) {
                double rate = data.production_rates[p * nm + m];
                if (rate <= 1e-6) continue;
                double prod_time = data.demands[p] / rate;
                int seq_len = machine_seqs[m].size();
                for (int pos = 0; pos <= seq_len; ++pos) {
                    int prev = (pos==0)? data.initial_state[m] : machine_seqs[m][pos-1];
                    int next = (pos<seq_len)? machine_seqs[m][pos] : -1;
                    double c_add = data.setup_costs[idx(prev, p, m)];
                    double c_rem = (next!=-1)? data.setup_costs[idx(prev,next,m)] : 0;
                    if(next!=-1) c_add += data.setup_costs[idx(p, next, m)];
                    double d_cost = c_add - c_rem;
                    
                    double avg_future_cost = 0.0;
                    if (pos == seq_len && remaining > 1) {
                        double sum_future = 0.0;
                        int count_future = 0;
                        for (int q = 0; q < np; ++q) {
                            if (!is_allocated[q] && q != p) {
                                sum_future += data.setup_costs[idx(p, q, m)];
                                count_future++;
                            }
                        }
                        if (count_future > 0) {
                            avg_future_cost = sum_future / count_future;
                        }
                    }

                    double FO = d_cost + (beta * avg_future_cost);

                    double t_add = data.setup_times[idx(prev, p, m)];
                    double t_rem = (next!=-1)? data.setup_times[idx(prev,next,m)] : 0;
                    if(next!=-1) t_add += data.setup_times[idx(p, next, m)];
                    double d_time = t_add + prod_time - t_rem;

                    if (machine_loads[m] + d_time <= data.machine_capacities[m]) {
                        if (FO < best_FO) {
                            best_FO = FO;
                            best_cost = d_cost; // Storing actual cost
                            best_m = m; best_pos = pos; best_time = d_time;
                            best_p = p;
                        }
                    }
                }
            }
        }

        if (best_p != -1) {
            machine_seqs[best_m].insert(machine_seqs[best_m].begin()+best_pos, best_p);
            machine_loads[best_m] += best_time;
            is_allocated[best_p] = true;
            total_FO_tracked += best_FO; // Acumulando a FO
            allocated_count++;
        } else {
            int panic_p = candidates[0];
            is_allocated[panic_p] = true;
            allocated_count++;
        }
    }

    printf("[CONFIRMAÇÃO] Fim da reconstrução. FO Total Acumulada (Custo Real + Penalidade Futura) = %.2f\n", total_FO_tracked);
    printf("[");
    for (int m = 0; m < nm; ++m) {
        printf("[");
        for (size_t i = 0; i < machine_seqs[m].size(); ++i) {
            printf("%d", machine_seqs[m][i]);
            if (i < machine_seqs[m].size() - 1) printf(", ");
        }
        printf("]");
        if (m < nm - 1) printf(", ");
    }
    printf("]\n");
}

#endif
