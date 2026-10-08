#ifndef _PROBLEM_H
#define _PROBLEM_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>

// 1. ESTRUTURA DE DADOS (Igual ao seu __init__ no Python)
struct TProblemData
{
    int n; // Tamanho do vetor de chaves (Obrigatório pelo RKO)

    int num_products;
    int num_machines;
    
    std::vector<double> machine_capacities; // T_r
    std::vector<int> initial_state;         // P0
    std::vector<double> demands;            // d_i
    
    // Matrizes linearizadas (vetor 1D) para performance
    // Acesso: [prod * num_machines + machine]
    std::vector<double> production_rates; 
    
    // Acesso: [prev_prod * num_products * num_machines + curr_prod * num_machines + machine]
    // Ou melhor: vetor de vetores para facilitar leitura, mas linear é mais rápido.
    // Vamos usar linearização: index = (prev * num_prod * num_mach) + (curr * num_mach) + mach
    std::vector<double> setup_costs;
    std::vector<double> setup_times;
    
    // Pre-computed: Average setup cost for each product (incompatibility metric)
    std::vector<double> avg_setup_costs;
};

// 2. LEITURA DE DADOS
void ReadData(char name[], TProblemData &data)
{
    std::ifstream file(name);
    if (!file.is_open()) {
        printf("\nERROR: File (%s) not found!\n", name);
        exit(1);
    }

    // Lê dimensões
    file >> data.num_products >> data.num_machines;
    data.n = data.num_products; // Tamanho do cromossomo

    // Lê capacidades
    data.machine_capacities.resize(data.num_machines);
    for(int i=0; i<data.num_machines; ++i) file >> data.machine_capacities[i];

    // Lê estado inicial
    data.initial_state.resize(data.num_machines);
    for(int i=0; i<data.num_machines; ++i) file >> data.initial_state[i];

    // Lê demandas
    data.demands.resize(data.num_products);
    for(int i=0; i<data.num_products; ++i) file >> data.demands[i];

    // Lê taxas de produção
    data.production_rates.resize(data.num_products * data.num_machines);
    for(int i=0; i<data.num_products; ++i) {
        for(int m=0; m<data.num_machines; ++m) {
            file >> data.production_rates[i * data.num_machines + m];
        }
    }

    // Lê custos de setup
    // Formato Li & Milne: Bloco por máquina
    data.setup_costs.resize(data.num_products * data.num_products * data.num_machines);
    std::string label;
    for(int m=0; m<data.num_machines; ++m) {
        file >> label; // "Reactor_Cost_X"
        for(int i=0; i<data.num_products; ++i) {
            for(int j=0; j<data.num_products; ++j) {
                // Indexação: [i][j][m]
                int idx = (i * data.num_products * data.num_machines) + (j * data.num_machines) + m;
                file >> data.setup_costs[idx];
            }
        }
    }

    // Lê tempos de setup
    data.setup_times.resize(data.num_products * data.num_products * data.num_machines);
    for(int m=0; m<data.num_machines; ++m) {
        file >> label; // "Reactor_Time_X"
        for(int i=0; i<data.num_products; ++i) {
            for(int j=0; j<data.num_products; ++j) {
                int idx = (i * data.num_products * data.num_machines) + (j * data.num_machines) + m;
                file >> data.setup_times[idx];
            }
        }
    }
    
    file.close();
    
    // Pre-calculate average setup cost for each product (look-ahead heuristic)
    // Higher value = product is harder to cluster (incompatible with others)
    data.avg_setup_costs.resize(data.num_products, 0.0);
    for (int i = 0; i < data.num_products; ++i) {
        double total = 0.0;
        int count = 0;
        for (int j = 0; j < data.num_products; ++j) {
            if (i == j) continue;
            for (int m = 0; m < data.num_machines; ++m) {
                // Cost from i to j
                int idx1 = (i * data.num_products * data.num_machines) + (j * data.num_machines) + m;
                // Cost from j to i
                int idx2 = (j * data.num_products * data.num_machines) + (i * data.num_machines) + m;
                total += data.setup_costs[idx1] + data.setup_costs[idx2];
                count += 2;
            }
        }
        data.avg_setup_costs[i] = (count > 0) ? (total / count) : 0.0;
    }
}

// ============================================================================
// 3. DECODER (Fitting Function with Look-Ahead Heuristic)
// ============================================================================
// Instead of greedy "min cost", uses Score = Priority / (ImmediateCost + β*FuturePenalty)
// Reference: Kierkosz & Łuczak (2019) Fitting Function concept

double Decoder(TSol &s, const TProblemData &data)
{
    // === FITTING FUNCTION PARAMETERS ===
    const double ALPHA = 10.0;   // Priority exponent (favor large demands)
    const double BETA = 1.0;    // Weight for future penalty (look-ahead)
    const double EPSILON = 1e-5; // Avoid division by zero
    
    const int np = data.num_products;
    const int nm = data.num_machines;
    
    // 1. Sort products by keys
    std::vector<int> sorted_products(np);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) {
        return s.rk[i] < s.rk[j];
    });

    // Solution structures
    std::vector<std::vector<int>> machine_seqs(nm);
    std::vector<double> machine_loads(nm, 0.0);
    double total_setup_cost = 0.0;
    double penalty = 0.0;

    // 2. Allocation with Fitting Function
    for (int p : sorted_products) {
        // Pre-compute priority for this product
        double priority = std::pow(data.demands[p], ALPHA);
        double future_penalty = data.avg_setup_costs[p];  // Look-ahead: incompatibility
        
        int best_machine = -1;
        int best_pos = -1;
        double max_score = -1e18;        // Maximize score (not minimize cost)
        double best_delta_cost = 0.0;
        double best_time_increase = 0.0;

        for (int m = 0; m < nm; ++m) {
            double rate = data.production_rates[p * nm + m];
            if (rate <= 1e-6) continue;

            double prod_time = data.demands[p] / rate;
            const auto& seq = machine_seqs[m];
            int seq_len = seq.size();

            // Find best position in this machine (Cheapest Insertion)
            double best_m_cost = 1e18;
            int best_m_pos = -1;
            double best_m_time = 0.0;

            for (int pos = 0; pos <= seq_len; ++pos) {
                int prev = (pos == 0) ? data.initial_state[m] : seq[pos - 1];
                int next = (pos < seq_len) ? seq[pos] : -1;

                // Index helper
                auto idx = [&](int i, int j) {
                    return (i * np * nm) + (j * nm) + m;
                };

                // Delta cost
                double cost_add = data.setup_costs[idx(prev, p)];
                double cost_rem = 0.0;
                if (next != -1) {
                    cost_add += data.setup_costs[idx(p, next)];
                    cost_rem = data.setup_costs[idx(prev, next)];
                }
                double delta_cost = cost_add - cost_rem;

                // Delta time
                double time_add = data.setup_times[idx(prev, p)];
                double time_rem = 0.0;
                if (next != -1) {
                    time_add += data.setup_times[idx(p, next)];
                    time_rem = data.setup_times[idx(prev, next)];
                }
                double delta_time = time_add + prod_time - time_rem;

                // Check capacity
                if (machine_loads[m] + delta_time <= data.machine_capacities[m]) {
                    if (delta_cost < best_m_cost) {
                        best_m_cost = delta_cost;
                        best_m_pos = pos;
                        best_m_time = delta_time;
                    }
                }
            }

            // If we found a valid position, compute Fitting Score
            if (best_m_pos != -1) {
                // Score = Priority / (ImmediateCost + β*FuturePenalty + ε)
                double immediate_cost = std::max(best_m_cost, 0.0) + EPSILON;
                double denominator = immediate_cost + BETA * future_penalty;
                double score = priority / denominator;

                if (score > max_score) {
                    max_score = score;
                    best_machine = m;
                    best_pos = best_m_pos;
                    best_delta_cost = best_m_cost;
                    best_time_increase = best_m_time;
                }
            }
        }

        // Perform allocation
        if (best_machine != -1) {
            machine_seqs[best_machine].insert(
                machine_seqs[best_machine].begin() + best_pos, p);
            machine_loads[best_machine] += best_time_increase;
            total_setup_cost += best_delta_cost;
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
    data.avg_setup_costs.clear();
}

// Função auxiliar para imprimir a solução na tela (C++ -> Python check)
void PrintSolution(TSol &s, const TProblemData &data)
{
    // 1. Reconstrói a solução (mesma lógica do Decoder)
    std::vector<int> sorted_products(data.n);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) {
        return s.rk[i] < s.rk[j];
    });

    std::vector<std::vector<int>> machine_seqs(data.num_machines);
    std::vector<double> machine_loads(data.num_machines, 0.0);

    for (int product_idx : sorted_products) {
        int best_machine = -1;
        int best_pos = -1;
        double min_delta_cost = std::numeric_limits<double>::infinity();
        double best_time_increase = 0.0;

        for (int m = 0; m < data.num_machines; ++m) {
            double rate = data.production_rates[product_idx * data.num_machines + m];
            if (rate <= 1e-6) continue;
            double prod_time = data.demands[product_idx] / rate;
            const auto& current_seq = machine_seqs[m];
            int seq_len = current_seq.size();

            for (int pos = 0; pos <= seq_len; ++pos) {
                int prev_prod = (pos == 0) ? data.initial_state[m] : current_seq[pos - 1];
                int next_prod = (pos < seq_len) ? current_seq[pos] : -1;

                int idx_prev_curr = (prev_prod * data.num_products * data.num_machines) + (product_idx * data.num_machines) + m;
                double cost_add = data.setup_costs[idx_prev_curr];
                double cost_rem = 0.0;

                if (next_prod != -1) {
                    int idx_curr_next = (product_idx * data.num_products * data.num_machines) + (next_prod * data.num_machines) + m;
                    int idx_prev_next = (prev_prod * data.num_products * data.num_machines) + (next_prod * data.num_machines) + m;
                    cost_add += data.setup_costs[idx_curr_next];
                    cost_rem = data.setup_costs[idx_prev_next];
                }
                double delta_cost = cost_add - cost_rem;

                int idx_time_prev_curr = (prev_prod * data.num_products * data.num_machines) + (product_idx * data.num_machines) + m;
                double time_add = data.setup_times[idx_time_prev_curr];
                double time_rem = 0.0;

                if (next_prod != -1) {
                    int idx_time_curr_next = (product_idx * data.num_products * data.num_machines) + (next_prod * data.num_machines) + m;
                    int idx_time_prev_next = (prev_prod * data.num_products * data.num_machines) + (next_prod * data.num_machines) + m;
                    time_add += data.setup_times[idx_time_curr_next];
                    time_rem = data.setup_times[idx_time_prev_next];
                }
                double delta_time = time_add + prod_time - time_rem;

                if (machine_loads[m] + delta_time <= data.machine_capacities[m]) {
                    if (delta_cost < min_delta_cost) {
                        min_delta_cost = delta_cost;
                        best_machine = m;
                        best_pos = pos;
                        best_time_increase = delta_time;
                    }
                    else if (std::abs(delta_cost - min_delta_cost) < 1e-6) {
                        if (best_machine != -1) {
                            double curr_rate = data.production_rates[product_idx * data.num_machines + best_machine];
                            if (rate > curr_rate) {
                                best_machine = m;
                                best_pos = pos;
                                best_time_increase = delta_time;
                            }
                        }
                    }
                }
            }
        }
        if (best_machine != -1) {
            machine_seqs[best_machine].insert(machine_seqs[best_machine].begin() + best_pos, product_idx);
            machine_loads[best_machine] += best_time_increase;
        }
    }

    // 2. IMPRIME A SEQUENCIA FORMATADA (Python-friendly)
    printf("\n=== SEQUENCE_START ===\n");
    printf("[");
    for (int m = 0; m < data.num_machines; ++m) {
        printf("[");
        for (size_t i = 0; i < machine_seqs[m].size(); ++i) {
            printf("%d", machine_seqs[m][i]);
            if (i < machine_seqs[m].size() - 1) printf(", ");
        }
        printf("]");
        if (m < data.num_machines - 1) printf(", ");
    }
    printf("]\n");
    printf("=== SEQUENCE_END ===\n");
}

#endif