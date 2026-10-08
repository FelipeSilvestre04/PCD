#ifndef _PROBLEM_BESTFIT_H
#define _PROBLEM_BESTFIT_H

// ============================================================================
// DECODER: BEST FIT (Cheapest Insertion)
// Testa TODAS as posições em TODAS as máquinas para cada produto
// ============================================================================

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>

// 1. ESTRUTURA DE DADOS
struct TProblemData
{
    int n; // Tamanho do vetor de chaves (Obrigatório pelo RKO)

    int num_products;
    int num_machines;
    
    std::vector<double> machine_capacities; // T_r
    std::vector<int> initial_state;         // P0
    std::vector<double> demands;            // d_i
    
    // Matrizes linearizadas (vetor 1D) para performance
    std::vector<double> production_rates; 
    std::vector<double> setup_costs;
    std::vector<double> setup_times;

    // Hard time limit for decoder (set from main.cpp)
    double start_time = 0.0;
    int max_time = 0;
};

// 2. LEITURA DE DADOS
void ReadData(char name[], TProblemData &data)
{
    std::ifstream file(name);
    if (!file.is_open()) {
        printf("\nERROR: File (%s) not found!\n", name);
        exit(1);
    }

    file >> data.num_products >> data.num_machines;
    data.n = data.num_products;

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

// ============================================================================
// 3. DECODER: BEST FIT (Cheapest Insertion - All Positions)
// ============================================================================
double DecoderImpl(TSol &s, const TProblemData &data)
{
    const int np = data.num_products;
    const int nm = data.num_machines;
    
    // 1. Sort products by keys
    std::vector<int> sorted_products(np);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) {
        return s.rk[i] == s.rk[j] ? i < j : s.rk[i] < s.rk[j];
    });

    // Solution structures
    std::vector<std::vector<int>> machine_seqs(nm);
    std::vector<double> machine_loads(nm, 0.0);
    double total_setup_cost = 0.0;
    double penalty = 0.0;

    // Index helper lambda
    auto idx = [np, nm](int prev, int curr, int m) {
        return (prev * np * nm) + (curr * nm) + m;
    };

    // 2. Greedy Allocation: Cheapest Insertion (all positions)
    for (int p : sorted_products) {
        int best_machine = -1;
        int best_pos = -1;
        double min_delta_cost = std::numeric_limits<double>::infinity();
        double best_time_increase = 0.0;

        for (int m = 0; m < nm; ++m) {
            double rate = data.production_rates[p * nm + m];
            if (rate <= 1e-6) continue;

            double prod_time = data.demands[p] / rate;
            const auto& seq = machine_seqs[m];
            int seq_len = seq.size();

            // Test ALL positions (0 to seq_len) - Cheapest Insertion
            for (int pos = 0; pos <= seq_len; ++pos) {
                int prev = (pos == 0) ? data.initial_state[m] : seq[pos - 1];
                int next = (pos < seq_len) ? seq[pos] : -1;

                // Delta cost
                double cost_add = data.setup_costs[idx(prev, p, m)];
                double cost_rem = 0.0;
                if (next != -1) {
                    cost_add += data.setup_costs[idx(p, next, m)];
                    cost_rem = data.setup_costs[idx(prev, next, m)];
                }
                double delta_cost = cost_add - cost_rem;

                // Delta time
                double time_add = data.setup_times[idx(prev, p, m)];
                double time_rem = 0.0;
                if (next != -1) {
                    time_add += data.setup_times[idx(p, next, m)];
                    time_rem = data.setup_times[idx(prev, next, m)];
                }
                double delta_time = time_add + prod_time - time_rem;

                // Check capacity and update best
                if (machine_loads[m] + delta_time <= data.machine_capacities[m]) {
                    if (delta_cost < min_delta_cost) {
                        min_delta_cost = delta_cost;
                        best_machine = m;
                        best_pos = pos;
                        best_time_increase = delta_time;
                    } else if (std::abs(delta_cost - min_delta_cost) < 1e-6 && best_machine != -1) {
                        if (rate > data.production_rates[p * nm + best_machine]) {
                            best_machine = m;
                            best_pos = pos;
                            best_time_increase = delta_time;
                        }
                    }
                }
            }
        }

        // Perform allocation
        if (best_machine != -1) {
            machine_seqs[best_machine].insert(
                machine_seqs[best_machine].begin() + best_pos, p);
            machine_loads[best_machine] += best_time_increase;
            total_setup_cost += min_delta_cost;
        } else {
            penalty += 100000.0 + (data.demands[p] * 1000.0);
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

// PrintSolution: Reconstruct and print the solution (for debugging/verification)
void PrintSolution(TSol &s, const TProblemData &data)
{
    const int np = data.num_products;
    const int nm = data.num_machines;
    
    std::vector<int> sorted_products(np);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) {
        return s.rk[i] == s.rk[j] ? i < j : s.rk[i] < s.rk[j];
    });

    std::vector<std::vector<int>> machine_seqs(nm);
    std::vector<double> machine_loads(nm, 0.0);

    auto idx = [np, nm](int prev, int curr, int m) {
        return (prev * np * nm) + (curr * nm) + m;
    };

    for (int p : sorted_products) {
        int best_machine = -1;
        int best_pos = -1;
        double min_delta_cost = std::numeric_limits<double>::infinity();
        double best_time_increase = 0.0;

        for (int m = 0; m < nm; ++m) {
            double rate = data.production_rates[p * nm + m];
            if (rate <= 1e-6) continue;
            double prod_time = data.demands[p] / rate;
            const auto& seq = machine_seqs[m];
            int seq_len = seq.size();

            for (int pos = 0; pos <= seq_len; ++pos) {
                int prev = (pos == 0) ? data.initial_state[m] : seq[pos - 1];
                int next = (pos < seq_len) ? seq[pos] : -1;

                double cost_add = data.setup_costs[idx(prev, p, m)];
                double cost_rem = 0.0;
                if (next != -1) {
                    cost_add += data.setup_costs[idx(p, next, m)];
                    cost_rem = data.setup_costs[idx(prev, next, m)];
                }
                double delta_cost = cost_add - cost_rem;

                double time_add = data.setup_times[idx(prev, p, m)];
                double time_rem = 0.0;
                if (next != -1) {
                    time_add += data.setup_times[idx(p, next, m)];
                    time_rem = data.setup_times[idx(prev, next, m)];
                }
                double delta_time = time_add + prod_time - time_rem;

                if (machine_loads[m] + delta_time <= data.machine_capacities[m]) {
                    if (delta_cost < min_delta_cost) {
                        min_delta_cost = delta_cost;
                        best_machine = m;
                        best_pos = pos;
                        best_time_increase = delta_time;
                    }
                }
            }
        }
        if (best_machine != -1) {
            machine_seqs[best_machine].insert(machine_seqs[best_machine].begin() + best_pos, p);
            machine_loads[best_machine] += best_time_increase;
        }
    }

    printf("\n=== SEQUENCE_START ===\n");
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
    printf("=== SEQUENCE_END ===\n");
}

#endif
