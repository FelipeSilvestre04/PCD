#ifndef _PROBLEM_2N_H
#define _PROBLEM_2N_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>

// ============================================================================
// DECODER 2N: Machine Assignment via Second Half of Chromosome
// ============================================================================
// Chromosome: [order_keys(0..N-1), machine_keys(N..2N-1)]
//   - First N keys: define product processing order (sorted ascending)
//   - Second N keys: define target machine for each product
//     machine = floor(key[N+p] * num_machines)
//   - Within target machine: Cheapest Insertion (all positions)
//   - Fallback: if target machine has no capacity, try all machines
// ============================================================================

struct TProblemData
{
    int n; // Chromosome size = 2 * num_products

    int num_products;
    int num_machines;
    
    std::vector<double> machine_capacities;
    std::vector<int> initial_state;
    std::vector<double> demands;
    std::vector<double> production_rates; 
    std::vector<double> setup_costs;
    std::vector<double> setup_times;
    std::vector<double> avg_setup_costs;
};

void ReadData(char name[], TProblemData &data)
{
    std::ifstream file(name);
    if (!file.is_open()) {
        printf("\nERROR: File (%s) not found!\n", name);
        exit(1);
    }

    file >> data.num_products >> data.num_machines;
    
    // KEY CHANGE: chromosome is 2N (N for order + N for machine assignment)
    data.n = 2 * data.num_products;

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
    
    // Pre-calculate average setup costs (not used in this decoder, but kept for compatibility)
    data.avg_setup_costs.resize(data.num_products, 0.0);
    for (int i = 0; i < data.num_products; ++i) {
        double total = 0.0;
        int count = 0;
        for (int j = 0; j < data.num_products; ++j) {
            if (i == j) continue;
            for (int m = 0; m < data.num_machines; ++m) {
                int idx1 = (i * data.num_products * data.num_machines) + (j * data.num_machines) + m;
                int idx2 = (j * data.num_products * data.num_machines) + (i * data.num_machines) + m;
                total += data.setup_costs[idx1] + data.setup_costs[idx2];
                count += 2;
            }
        }
        data.avg_setup_costs[i] = (count > 0) ? (total / count) : 0.0;
    }
}

// ============================================================================
// Helper: Find best insertion position for product p on machine m
// Returns: delta_cost, delta_time, best_pos  (pos=-1 if infeasible)
// ============================================================================
struct InsertionResult {
    double delta_cost;
    double delta_time;
    int pos;
};

InsertionResult FindBestInsertion(int p, int m,
                                  const std::vector<std::vector<int>>& machine_seqs,
                                  const std::vector<double>& machine_loads,
                                  const TProblemData& data)
{
    InsertionResult best = {1e18, 0.0, -1};
    
    const int np = data.num_products;
    const int nm = data.num_machines;
    
    double rate = data.production_rates[p * nm + m];
    if (rate <= 1e-6) return best;
    
    double prod_time = data.demands[p] / rate;
    const auto& seq = machine_seqs[m];
    int seq_len = seq.size();
    
    auto idx = [&](int i, int j) {
        return (i * np * nm) + (j * nm) + m;
    };
    
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
            if (delta_cost < best.delta_cost) {
                best.delta_cost = delta_cost;
                best.delta_time = delta_time;
                best.pos = pos;
            }
        }
    }
    
    return best;
}

// ============================================================================
// DECODER 2N
// ============================================================================
double Decoder(TSol &s, const TProblemData &data)
{
    const int np = data.num_products;
    const int nm = data.num_machines;
    
    // 1. Sort products by first N keys (order keys)
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

    // 2. Allocate each product
    for (int p : sorted_products) {
        // Determine target machine from second half of chromosome
        double machine_key = s.rk[np + p];  // key in [0, 1)
        int target_machine = (int)(machine_key * nm);
        if (target_machine >= nm) target_machine = nm - 1;
        if (target_machine < 0) target_machine = 0;
        
        // Try target machine first (cheapest insertion within it)
        InsertionResult best = FindBestInsertion(p, target_machine, machine_seqs, machine_loads, data);
        int best_machine = (best.pos != -1) ? target_machine : -1;
        
        // If target machine failed, try all other machines
        if (best_machine == -1) {
            for (int m = 0; m < nm; ++m) {
                if (m == target_machine) continue;
                InsertionResult res = FindBestInsertion(p, m, machine_seqs, machine_loads, data);
                if (res.pos != -1 && res.delta_cost < best.delta_cost) {
                    best = res;
                    best_machine = m;
                }
            }
        }
        
        // Perform allocation
        if (best_machine != -1) {
            machine_seqs[best_machine].insert(
                machine_seqs[best_machine].begin() + best.pos, p);
            machine_loads[best_machine] += best.delta_time;
            total_setup_cost += best.delta_cost;
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

// PrintSolution: Reconstruct and print the solution
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
        double machine_key = s.rk[np + p];
        int target_machine = (int)(machine_key * nm);
        if (target_machine >= nm) target_machine = nm - 1;
        if (target_machine < 0) target_machine = 0;
        
        InsertionResult best = FindBestInsertion(p, target_machine, machine_seqs, machine_loads, data);
        int best_machine = (best.pos != -1) ? target_machine : -1;
        
        if (best_machine == -1) {
            for (int m = 0; m < nm; ++m) {
                if (m == target_machine) continue;
                InsertionResult res = FindBestInsertion(p, m, machine_seqs, machine_loads, data);
                if (res.pos != -1 && res.delta_cost < best.delta_cost) {
                    best = res;
                    best_machine = m;
                }
            }
        }
        
        if (best_machine != -1) {
            machine_seqs[best_machine].insert(
                machine_seqs[best_machine].begin() + best.pos, p);
            machine_loads[best_machine] += best.delta_time;
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
