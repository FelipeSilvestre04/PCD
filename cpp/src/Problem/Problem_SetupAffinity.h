#ifndef _PROBLEM_SETUP_AFFINITY_H
#define _PROBLEM_SETUP_AFFINITY_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>

// ============================================================================
// DECODER: Setup Affinity Machine Selection
// ============================================================================
// Machine selection: choose machine where product p has the LOWEST total
// setup affinity (sum of row + column in the setup cost matrix for that machine).
//
// Affinity(p, m) = Σ_j setup_cost[p][j][m] + Σ_j setup_cost[j][p][m]
//
// This measures how "compatible" product p is with machine m globally.
// A lower affinity means p has cheaper setups overall on machine m.
//
// Within the chosen machine: Cheapest Insertion (all positions).
// ============================================================================

struct TProblemData
{
    int n; // Chromosome size = num_products

    int num_products;
    int num_machines;
    
    std::vector<double> machine_capacities;
    std::vector<int> initial_state;
    std::vector<double> demands;
    std::vector<double> production_rates; 
    std::vector<double> setup_costs;
    std::vector<double> setup_times;
    std::vector<double> avg_setup_costs;
    
    // Pre-computed: affinity[p * num_machines + m] = row+col sum for product p on machine m
    std::vector<double> affinity;
};

void ReadData(char name[], TProblemData &data)
{
    std::ifstream file(name);
    if (!file.is_open()) {
        printf("\nERROR: File (%s) not found!\n", name);
        exit(1);
    }

    file >> data.num_products >> data.num_machines;
    data.n = data.num_products; // Standard N chromosome

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
    
    // Pre-compute setup affinity for each (product, machine) pair
    // affinity(p, m) = sum of row p on machine m + sum of column p on machine m
    const int np = data.num_products;
    const int nm = data.num_machines;
    
    data.affinity.resize(np * nm, 0.0);
    for (int p = 0; p < np; ++p) {
        for (int m = 0; m < nm; ++m) {
            double row_sum = 0.0; // costs FROM p TO all others on machine m
            double col_sum = 0.0; // costs FROM all others TO p on machine m
            for (int j = 0; j < np; ++j) {
                if (j == p) continue;
                // p -> j (row)
                int idx_pj = (p * np * nm) + (j * nm) + m;
                row_sum += data.setup_costs[idx_pj];
                // j -> p (column)
                int idx_jp = (j * np * nm) + (p * nm) + m;
                col_sum += data.setup_costs[idx_jp];
            }
            data.affinity[p * nm + m] = row_sum + col_sum;
        }
    }
    
    // Also compute avg_setup_costs for compatibility
    data.avg_setup_costs.resize(np, 0.0);
    for (int i = 0; i < np; ++i) {
        double total = 0.0;
        int count = 0;
        for (int j = 0; j < np; ++j) {
            if (i == j) continue;
            for (int m = 0; m < nm; ++m) {
                int idx1 = (i * np * nm) + (j * nm) + m;
                int idx2 = (j * np * nm) + (i * nm) + m;
                total += data.setup_costs[idx1] + data.setup_costs[idx2];
                count += 2;
            }
        }
        data.avg_setup_costs[i] = (count > 0) ? (total / count) : 0.0;
    }
}

// ============================================================================
// DECODER: Setup Affinity
// ============================================================================
double Decoder(TSol &s, const TProblemData &data)
{
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

    // 2. Rank machines by affinity for each product, then try in that order
    for (int p : sorted_products) {
        // Sort machines by affinity (lowest = best fit for this product)
        std::vector<int> machine_order(nm);
        std::iota(machine_order.begin(), machine_order.end(), 0);
        std::sort(machine_order.begin(), machine_order.end(), [&](int a, int b) {
            return data.affinity[p * nm + a] < data.affinity[p * nm + b];
        });
        
        int best_machine = -1;
        int best_pos = -1;
        double best_delta_cost = 1e18;
        double best_time_increase = 0.0;
        
        // Try machines in affinity order: first valid insertion wins
        for (int mi = 0; mi < nm; ++mi) {
            int m = machine_order[mi];
            
            double rate = data.production_rates[p * nm + m];
            if (rate <= 1e-6) continue;

            double prod_time = data.demands[p] / rate;
            const auto& seq = machine_seqs[m];
            int seq_len = seq.size();

            auto idx = [&](int i, int j) {
                return (i * np * nm) + (j * nm) + m;
            };

            // Find cheapest insertion position on this machine
            double best_m_cost = 1e18;
            int best_m_pos = -1;
            double best_m_time = 0.0;

            for (int pos = 0; pos <= seq_len; ++pos) {
                int prev = (pos == 0) ? data.initial_state[m] : seq[pos - 1];
                int next = (pos < seq_len) ? seq[pos] : -1;

                double cost_add = data.setup_costs[idx(prev, p)];
                double cost_rem = 0.0;
                if (next != -1) {
                    cost_add += data.setup_costs[idx(p, next)];
                    cost_rem = data.setup_costs[idx(prev, next)];
                }
                double delta_cost = cost_add - cost_rem;

                double time_add = data.setup_times[idx(prev, p)];
                double time_rem = 0.0;
                if (next != -1) {
                    time_add += data.setup_times[idx(p, next)];
                    time_rem = data.setup_times[idx(prev, next)];
                }
                double delta_time = time_add + prod_time - time_rem;

                if (machine_loads[m] + delta_time <= data.machine_capacities[m]) {
                    if (delta_cost < best_m_cost) {
                        best_m_cost = delta_cost;
                        best_m_pos = pos;
                        best_m_time = delta_time;
                    }
                }
            }

            // If found a valid position, use this machine (best affinity with valid insertion)
            if (best_m_pos != -1) {
                best_machine = m;
                best_pos = best_m_pos;
                best_delta_cost = best_m_cost;
                best_time_increase = best_m_time;
                break; // Take the first machine with lowest affinity that fits
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
    data.affinity.clear();
}

// PrintSolution
void PrintSolution(TSol &s, const TProblemData &data)
{
    const int np = data.num_products;
    const int nm = data.num_machines;
    
    std::vector<int> sorted_products(np);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) {
        return s.rk[i] < s.rk[j];
    });

    std::vector<std::vector<int>> machine_seqs(nm);
    std::vector<double> machine_loads(nm, 0.0);

    for (int p : sorted_products) {
        std::vector<int> machine_order(nm);
        std::iota(machine_order.begin(), machine_order.end(), 0);
        std::sort(machine_order.begin(), machine_order.end(), [&](int a, int b) {
            return data.affinity[p * nm + a] < data.affinity[p * nm + b];
        });
        
        for (int mi = 0; mi < nm; ++mi) {
            int m = machine_order[mi];
            
            double rate = data.production_rates[p * nm + m];
            if (rate <= 1e-6) continue;

            double prod_time = data.demands[p] / rate;
            const auto& seq = machine_seqs[m];
            int seq_len = seq.size();

            auto idx = [&](int i, int j) {
                return (i * np * nm) + (j * nm) + m;
            };

            double best_m_cost = 1e18;
            int best_m_pos = -1;
            double best_m_time = 0.0;

            for (int pos = 0; pos <= seq_len; ++pos) {
                int prev = (pos == 0) ? data.initial_state[m] : seq[pos - 1];
                int next = (pos < seq_len) ? seq[pos] : -1;

                double cost_add = data.setup_costs[idx(prev, p)];
                double cost_rem = 0.0;
                if (next != -1) {
                    cost_add += data.setup_costs[idx(p, next)];
                    cost_rem = data.setup_costs[idx(prev, next)];
                }
                double delta_cost = cost_add - cost_rem;

                double time_add = data.setup_times[idx(prev, p)];
                double time_rem = 0.0;
                if (next != -1) {
                    time_add += data.setup_times[idx(p, next)];
                    time_rem = data.setup_times[idx(prev, next)];
                }
                double delta_time = time_add + prod_time - time_rem;

                if (machine_loads[m] + delta_time <= data.machine_capacities[m]) {
                    if (delta_cost < best_m_cost) {
                        best_m_cost = delta_cost;
                        best_m_pos = pos;
                        best_m_time = delta_time;
                    }
                }
            }

            if (best_m_pos != -1) {
                machine_seqs[m].insert(machine_seqs[m].begin() + best_m_pos, p);
                machine_loads[m] += best_m_time;
                break;
            }
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
