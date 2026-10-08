#ifndef _PROBLEM_DYNAMICWINDOWGCI_V2_H
#define _PROBLEM_DYNAMICWINDOWGCI_V2_H

// ============================================================================
// DECODER V2: DYNAMIC WINDOW GCI + MACHINE WINDOWING
//
// Chromosome: size N + M + 2
//   Genes [0..N-1]:     Product Priorities
//   Genes [N..N+M-1]:   Machine Priorities
//   Gene  [N+M]:        Alpha (product window factor)
//   Gene  [N+M+1]:      Beta  (machine window factor)
//
// Logic:
// 1. Sort products by keys [0..N-1] (ascending = higher priority).
// 2. Sort machines by keys [N..N+M-1] (ascending = higher priority).
// 3. K_prod = max(1, ceil(alpha * remaining_products))
// 4. K_maq  = max(1, ceil(beta  * M))
// 5. For each product candidate (top K_prod), try insertion in top K_maq machines.
// 6. FALLBACK: if no candidate fits in K_maq machines, retry with ALL M machines.
// 7. If still no fit: penalty.
//
// beta ~ 0: only 1 machine tested   -> max diversification of allocation
// beta ~ 1: all machines tested     -> identical to V1
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

// 1. DATA STRUCTURE
struct TProblemData
{
    int n; // Chromosome size (N + M + 2)

    int num_products;
    int num_machines;

    std::vector<double> machine_capacities; // T_r
    std::vector<int> initial_state;         // P0
    std::vector<double> demands;            // d_i
    
    std::vector<double> production_rates; 
    std::vector<double> setup_costs;
    std::vector<double> setup_times;
    
    // Hard time limit for decoder (set from main.cpp)
    double start_time = 0.0;
    double max_time = 1e9;
};

// 2. READ DATA
void ReadData(char name[], TProblemData &data)
{
    std::ifstream file(name);
    if (!file.is_open()) {
        printf("\nERROR: File (%s) not found!\n", name);
        exit(1);
    }

    file >> data.num_products >> data.num_machines;
    
    // Chromosome: N products + M machines + alpha + beta
    data.n = data.num_products + data.num_machines + 2;

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
// Helper: try inserting candidates into a specific set of machines
// Returns true if at least one feasible insertion was found
// ============================================================================
static bool TryInsertCandidates(
    const std::vector<int>& candidates,
    const std::vector<int>& machines_to_try,
    const TProblemData& data,
    const std::vector<std::vector<int>>& machine_seqs,
    const std::vector<double>& machine_loads,
    int& out_best_p, int& out_best_m, int& out_best_pos,
    double& out_best_cost, double& out_best_time)
{
    const int np = data.num_products;
    const int nm = data.num_machines;
    
    auto idx = [np, nm](int prev, int curr, int m) {
        return (prev * np * nm) + (curr * nm) + m;
    };

    out_best_p = -1;
    out_best_m = -1;
    out_best_pos = -1;
    out_best_cost = std::numeric_limits<double>::infinity();
    out_best_time = 0.0;

    for (int p : candidates) {
        double local_best_cost = std::numeric_limits<double>::infinity();
        int local_best_m = -1;
        int local_best_pos = -1;
        double local_best_time = 0.0;

        for (int m : machines_to_try) {
            double rate = data.production_rates[p * nm + m];
            if (rate <= 1e-6) continue;

            double prod_time = data.demands[p] / rate;
            const auto& seq = machine_seqs[m];
            int seq_len = (int)seq.size();

            for (int pos = 0; pos <= seq_len; ++pos) {
                int prev = (pos == 0) ? data.initial_state[m] : seq[pos - 1];
                int next = (pos < seq_len) ? seq[pos] : -1;

                // Cost Delta
                double cost_add = data.setup_costs[idx(prev, p, m)];
                double cost_rem = 0.0;
                if (next != -1) {
                    cost_add += data.setup_costs[idx(p, next, m)];
                    cost_rem = data.setup_costs[idx(prev, next, m)];
                }
                double delta_cost = cost_add - cost_rem;

                // Time Delta
                double time_add = data.setup_times[idx(prev, p, m)];
                double time_rem = 0.0;
                if (next != -1) {
                    time_add += data.setup_times[idx(p, next, m)];
                    time_rem = data.setup_times[idx(prev, next, m)];
                }
                double delta_time = time_add + prod_time - time_rem;

                if (machine_loads[m] + delta_time <= data.machine_capacities[m]) {
                    if (delta_cost < local_best_cost) {
                        local_best_cost = delta_cost;
                        local_best_m = m;
                        local_best_pos = pos;
                        local_best_time = delta_time;
                    }
                }
            }
        }

        // Compare this product's best vs global best
        if (local_best_cost < out_best_cost) {
            out_best_cost = local_best_cost;
            out_best_time = local_best_time;
            out_best_p = p;
            out_best_m = local_best_m;
            out_best_pos = local_best_pos;
        }
    }

    return (out_best_p != -1);
}

// ============================================================================
// 3. DECODER
// ============================================================================
double Decoder(TSol &s, const TProblemData &data)
{
    const int np = data.num_products;
    const int nm = data.num_machines;
    
    // Extract alpha and beta from chromosome
    double alpha = s.rk[np + nm];       // Gene [N+M]
    double beta  = s.rk[np + nm + 1];   // Gene [N+M+1]
    
    // Sort products by keys [0..N-1]
    std::vector<int> sorted_products(np);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) {
        return s.rk[i] < s.rk[j];
    });

    // Sort machines by keys [N..N+M-1]
    std::vector<int> sorted_machines(nm);
    std::iota(sorted_machines.begin(), sorted_machines.end(), 0);
    std::sort(sorted_machines.begin(), sorted_machines.end(), [&](int a, int b) {
        return s.rk[np + a] < s.rk[np + b];
    });

    // Machine window size (fixed throughout the decode)
    int k_maq = std::max(1, (int)std::ceil(beta * nm));
    
    // Build machine window (top k_maq from sorted_machines)
    std::vector<int> machine_window(sorted_machines.begin(), sorted_machines.begin() + k_maq);
    
    // Build full machine list for fallback
    std::vector<int> all_machines(nm);
    std::iota(all_machines.begin(), all_machines.end(), 0);

    // Solution state
    std::vector<bool> is_allocated(np, false);
    int allocated_count = 0;
    std::vector<std::vector<int>> machine_seqs(nm);
    std::vector<double> machine_loads(nm, 0.0);
    double total_setup_cost = 0.0;
    double penalty = 0.0;

    while (allocated_count < np) {
        // Panic mode: if time expired, force k_prod=1 and use all machines
        bool panic_mode = (omp_get_wtime() - data.start_time >= data.max_time);

        int remaining = np - allocated_count;
        
        // Product window
        int k_prod = std::max(1, (int)std::ceil(alpha * remaining));
        if (panic_mode) k_prod = 1;
        
        // Collect top k_prod unallocated products
        std::vector<int> candidates;
        candidates.reserve(k_prod);
        for (int p : sorted_products) {
            if (!is_allocated[p]) {
                candidates.push_back(p);
                if ((int)candidates.size() == k_prod) break;
            }
        }

        int best_p, best_m, best_pos;
        double best_cost, best_time;

        // Try with machine window first
        bool found = false;
        if (!panic_mode) {
            found = TryInsertCandidates(candidates, machine_window, data,
                                        machine_seqs, machine_loads,
                                        best_p, best_m, best_pos, best_cost, best_time);
        }

        // FALLBACK: if nothing fit in the window, try ALL machines
        if (!found) {
            found = TryInsertCandidates(candidates, all_machines, data,
                                        machine_seqs, machine_loads,
                                        best_p, best_m, best_pos, best_cost, best_time);
        }

        // Allocate
        if (found) {
            machine_seqs[best_m].insert(machine_seqs[best_m].begin() + best_pos, best_p);
            machine_loads[best_m] += best_time;
            total_setup_cost += best_cost;
            is_allocated[best_p] = true;
            allocated_count++;
        } else {
            // No feasible insertion anywhere — penalty
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

// PrintSolution: reconstruct for logging
void PrintSolution(TSol &s, const TProblemData &data)
{
    const int np = data.num_products;
    const int nm = data.num_machines;
    
    double alpha = s.rk[np + nm];
    double beta  = s.rk[np + nm + 1];
    
    printf("\n[DynamicWindowGCI_v2 Alpha=%.4f Beta=%.4f]\n", alpha, beta);

    // Sort products
    std::vector<int> sorted_products(np);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) {
        return s.rk[i] < s.rk[j];
    });

    // Sort machines
    std::vector<int> sorted_machines(nm);
    std::iota(sorted_machines.begin(), sorted_machines.end(), 0);
    std::sort(sorted_machines.begin(), sorted_machines.end(), [&](int a, int b) {
        return s.rk[np + a] < s.rk[np + b];
    });

    int k_maq = std::max(1, (int)std::ceil(beta * nm));
    std::vector<int> machine_window(sorted_machines.begin(), sorted_machines.begin() + k_maq);
    std::vector<int> all_machines(nm);
    std::iota(all_machines.begin(), all_machines.end(), 0);

    std::vector<bool> is_allocated(np, false);
    int allocated_count = 0;
    std::vector<std::vector<int>> machine_seqs(nm);
    std::vector<double> machine_loads(nm, 0.0);

    while (allocated_count < np) {
        int remaining = np - allocated_count;
        int k_prod = std::max(1, (int)std::ceil(alpha * remaining));
        
        std::vector<int> candidates;
        candidates.reserve(k_prod);
        for (int p : sorted_products) {
            if (!is_allocated[p]) {
                candidates.push_back(p);
                if ((int)candidates.size() == k_prod) break;
            }
        }

        int best_p, best_m, best_pos;
        double best_cost, best_time;

        bool found = TryInsertCandidates(candidates, machine_window, data,
                                          machine_seqs, machine_loads,
                                          best_p, best_m, best_pos, best_cost, best_time);
        if (!found) {
            found = TryInsertCandidates(candidates, all_machines, data,
                                          machine_seqs, machine_loads,
                                          best_p, best_m, best_pos, best_cost, best_time);
        }

        if (found) {
            machine_seqs[best_m].insert(machine_seqs[best_m].begin() + best_pos, best_p);
            machine_loads[best_m] += best_time;
            is_allocated[best_p] = true;
            allocated_count++;
        } else {
            int panic_p = candidates[0];
            is_allocated[panic_p] = true;
            allocated_count++;
        }
    }

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
