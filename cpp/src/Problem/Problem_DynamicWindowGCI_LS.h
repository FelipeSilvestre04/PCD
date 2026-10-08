#ifndef _PROBLEM_DYNAMICWINDOWGCI_H
#define _PROBLEM_DYNAMICWINDOWGCI_H

// ============================================================================
// DECODER: DYNAMIC WINDOW GLOBAL CHEAPEST INSERTION (Hybrid)
//
// Chromosome: size N + 1
// Genes 0..N-1: Product Priorities (Topological Urgency)
// Gene N: Window Factor Alpha (0.0 <= Alpha <= 1.0)
//
// Logic:
// 1. Sort all unallocated products by priority (Gene value: High > Low).
// 2. Determine window size K = max(1, ceil(Alpha * RemainingProducts)).
// 3. Evaluate Cheapest Insertion cost for the TOP K products in the sorted list.
// 4. Select the best (lowest cost) insertion among these K candidates.
// 5. Allocate and repeat.
//
// Behavior:
// - Alpha ~ 0: Pure Insertion Sort (Greedy on Order only). Good for feasibility.
// - Alpha ~ 1: Pure Global CI (Greedy on Cost only). Good for minimizing setup.
// - Intermediate Alpha: Balance between forcing hard items early and minimizing cost.
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
    int n; // Chromosome size (N + 1)

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
    double max_time = 1e9;  // default: no limit
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
    
    // IMPORTANT: Chromosome size is N + 1 (N priorities + 1 alpha)
    data.n = data.num_products + 1;

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
// 3. DECODER
// ============================================================================
double Decoder(TSol &s, const TProblemData &data)
{
    const int np = data.num_products;
    const int nm = data.num_machines;
    
    // 1. Extract Alpha from the last gene
    double alpha = s.rk[np]; // Gene index N (0-based is np)
    
    // 2. Sort all products by Priority (Genes 0 to N-1)
    // High key value = High Priority (processed earlier in the candidate list)
    // Actually, RKO usually min-sorts. Let's assume lower key = processed earlier in list.
    std::vector<int> sorted_products(np);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) {
        return s.rk[i] < s.rk[j]; // Ascending sort: smaller key first in list
    });

    // Valid list management
    // Instead of removing from vector (expensive), we use a bool mask and iterate
    std::vector<bool> is_allocated(np, false);
    int allocated_count = 0;

    // Solution state
    std::vector<std::vector<int>> machine_seqs(nm);
    std::vector<double> machine_loads(nm, 0.0);
    double total_setup_cost = 0.0;
    double penalty = 0.0;

    auto idx = [np, nm](int prev, int curr, int m) {
        return (prev * np * nm) + (curr * nm) + m;
    };

    while (allocated_count < np) {
        // Panic Mode: if time limit exceeded, switch to Greedy (k=1) to finish fast
        bool panic_mode = false;
        if (omp_get_wtime() - data.start_time >= data.max_time) {
            panic_mode = true;
        }

        int remaining = np - allocated_count;
        
        // 3. Determine Window Size K
        int k = std::max(1, (int)std::ceil(alpha * remaining));
        
        // If panic mode, force greedy (k=1)
        if (panic_mode) {
             k = 1;
        }
        // Else, no cap (unlimited window) as requested
        // else {
        //    const int MAX_WINDOW = 50;
        //    if (k > MAX_WINDOW) k = MAX_WINDOW;
        // }
        
        // 4. Identify Top K unallocated candidates
        // iterate through sorted_products and pick first k unallocated
        std::vector<int> candidates;
        candidates.reserve(k);
        for (int p : sorted_products) {
            if (!is_allocated[p]) {
                candidates.push_back(p);
                if ((int)candidates.size() == k) break;
            }
        }

        // 5. Evaluate GCI for candidates
        int best_p = -1;
        int best_m = -1;
        int best_pos = -1;
        double best_increase_cost = std::numeric_limits<double>::infinity();
        double best_increase_time = 0.0;

        for (int p : candidates) {
            
            // Find best insertion for this specific product p
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

            // Compare local best for product p against global best
            if (local_best_cost < best_increase_cost) {
                best_increase_cost = local_best_cost;
                best_increase_time = local_best_time;
                best_p = p;
                best_m = local_best_m;
                best_pos = local_best_pos;
            }
        } // end candidates loop

        // 6. Allocate best
        if (best_p != -1) {
            machine_seqs[best_m].insert(machine_seqs[best_m].begin() + best_pos, best_p);
            machine_loads[best_m] += best_increase_time;
            total_setup_cost += best_increase_cost;
            is_allocated[best_p] = true;
            allocated_count++;
        } else {
            // Cannot allocate any of the K candidates?
            // Fallback: Pick the first one from candidates (highest priority) 
            // and force allocate with penalty to one valid machine, or huge penalty if impossible.
            // Simplified: Just take the first candidate and penalize heavily.
            int panic_p = candidates[0];
            penalty += 1e9 + data.demands[panic_p] * 1000.0;
            is_allocated[panic_p] = true;
            allocated_count++;
        }
    }

    // =======================================================
    // LOCAL SEARCH: IMI with Speed-Up (Worst-Edge focused)
    // Ref: Vallada and Ruiz (2011) Speed-up procedure
    // Contains Shift and Swap to break capacity thresholds
    // =======================================================
    if (penalty == 0.0) { // Only refine feasible solutions
        int ls_iters = 0;
        bool improved = true;
        while (improved && ls_iters < 50) {
            improved = false;
            ls_iters++;

            // 1. Calculate marginal costs
            struct EdgeInfo { double mc; int p; int m; int pos; };
            std::vector<EdgeInfo> edges;
            edges.reserve(np);

            for (int m = 0; m < nm; ++m) {
                int sl = machine_seqs[m].size();
                for (int pos = 0; pos < sl; ++pos) {
                    int p = machine_seqs[m][pos];
                    int prev = (pos == 0) ? data.initial_state[m] : machine_seqs[m][pos - 1];
                    int next = (pos < sl - 1) ? machine_seqs[m][pos + 1] : -1;
                    
                    double cost_in = data.setup_costs[idx(prev, p, m)];
                    double cost_out = (next != -1) ? data.setup_costs[idx(p, next, m)] : 0.0;
                    double cost_bp = (next != -1) ? data.setup_costs[idx(prev, next, m)] : 0.0;
                    
                    edges.push_back({cost_in + cost_out - cost_bp, p, m, pos});
                }
            }

            // 2. Sort descending
            std::sort(edges.begin(), edges.end(), [](const EdgeInfo& a, const EdgeInfo& b) {
                return a.mc > b.mc;
            });

            // 3. Test shifting the top W worst jobs
            int limit = std::min(5, (int)edges.size()); // User requested top 5 worst edges
            for (int e = 0; e < limit; ++e) {
                int p1 = edges[e].p;
                int m1 = edges[e].m;
                int pos1 = edges[e].pos;

                int sl1 = machine_seqs[m1].size();
                int prev1 = (pos1 == 0) ? data.initial_state[m1] : machine_seqs[m1][pos1 - 1];
                int next1 = (pos1 < sl1 - 1) ? machine_seqs[m1][pos1 + 1] : -1;
                
                double cost_bp1 = (next1 != -1) ? data.setup_costs[idx(prev1, next1, m1)] : 0.0;
                double delta_rem = data.setup_costs[idx(prev1, p1, m1)] + ((next1 != -1) ? data.setup_costs[idx(p1, next1, m1)] : 0.0) - cost_bp1; // Cost saved by removing p1

                double pt1_m1 = data.demands[p1] / data.production_rates[p1 * nm + m1];
                double t_bp1 = (next1 != -1) ? data.setup_times[idx(prev1, next1, m1)] : 0.0;
                double time_relief_m1 = data.setup_times[idx(prev1, p1, m1)] + pt1_m1 + ((next1 != -1) ? data.setup_times[idx(p1, next1, m1)] : 0.0) - t_bp1;

                // Move 1: INTRA-MACHINE SHIFT (Reorder within the same machine)
                int best_intra_pos = -1;
                double best_intra_cost_diff = 0.0; // Needs to be strictly negative (improvement)

                if (sl1 > 1) { // Only makes sense if there are at least 2 jobs
                    // Physically simulate the removal for O(L) exact recalculation
                    std::vector<int> temp_seq = machine_seqs[m1];
                    temp_seq.erase(temp_seq.begin() + pos1);
                    
                    // Original cost of this machine
                    double orig_m1_cost = 0;
                    for (size_t i = 0; i < machine_seqs[m1].size(); ++i) {
                        int p = machine_seqs[m1][i];
                        int prev = (i == 0) ? data.initial_state[m1] : machine_seqs[m1][i-1];
                        orig_m1_cost += data.setup_costs[idx(prev, p, m1)];
                    }

                    for (size_t pos2 = 0; pos2 <= temp_seq.size(); ++pos2) {
                        if (pos2 == pos1) continue; // Same position
                        
                        temp_seq.insert(temp_seq.begin() + pos2, p1);
                        
                        double new_m1_cost = 0;
                        for (size_t i = 0; i < temp_seq.size(); ++i) {
                            int p = temp_seq[i];
                            int prev = (i == 0) ? data.initial_state[m1] : temp_seq[i-1];
                            new_m1_cost += data.setup_costs[idx(prev, p, m1)];
                        }
                        
                        double diff = new_m1_cost - orig_m1_cost;
                        if (diff < best_intra_cost_diff - 1e-4) { // Time doesn't change since it's the same machine
                            best_intra_cost_diff = diff;
                            best_intra_pos = pos2;
                        }
                        
                        temp_seq.erase(temp_seq.begin() + pos2); // backtrack
                    }
                }

                if (best_intra_pos != -1) {
                    total_setup_cost += best_intra_cost_diff;
                    machine_seqs[m1].erase(machine_seqs[m1].begin() + pos1);
                    machine_seqs[m1].insert(machine_seqs[m1].begin() + best_intra_pos, p1);
                    improved = true;
                    break;
                }

                // Move 2: INTER-MACHINE SHIFT (Insertion into other machines)
                int best_m2_shift = -1, best_pos2_shift = -1;
                double best_delta_add_shift = std::numeric_limits<double>::infinity();
                double best_time_add_shift = 0.0;

                for (int m2 = 0; m2 < nm; ++m2) {
                    if (m2 == m1) continue;
                    double rate = data.production_rates[p1 * nm + m2];
                    if (rate <= 1e-6) continue;
                    double pt2 = data.demands[p1] / rate;

                    int sl2 = machine_seqs[m2].size();
                    for (int pos2 = 0; pos2 <= sl2; ++pos2) {
                        int prev2 = (pos2 == 0) ? data.initial_state[m2] : machine_seqs[m2][pos2 - 1];
                        int next2 = (pos2 < sl2) ? machine_seqs[m2][pos2] : -1;

                        double cost_add = data.setup_costs[idx(prev2, p1, m2)];
                        double cost_rem_bp = (next2 != -1) ? data.setup_costs[idx(prev2, next2, m2)] : 0.0;
                        if (next2 != -1) cost_add += data.setup_costs[idx(p1, next2, m2)];
                        
                        double delta_add = cost_add - cost_rem_bp;
                        
                        if (delta_add < delta_rem - 1e-4) {
                            double time_add = data.setup_times[idx(prev2, p1, m2)];
                            double time_rem_bp = (next2 != -1) ? data.setup_times[idx(prev2, next2, m2)] : 0.0;
                            if (next2 != -1) time_add += data.setup_times[idx(p1, next2, m2)];
                            
                            double delta_time_add = time_add + pt2 - time_rem_bp;
                            
                            if (machine_loads[m2] + delta_time_add <= data.machine_capacities[m2]) {
                                if (delta_add < best_delta_add_shift) {
                                    best_delta_add_shift = delta_add;
                                    best_m2_shift = m2;
                                    best_pos2_shift = pos2;
                                    best_time_add_shift = delta_time_add;
                                }
                            }
                        }
                    }
                }

                if (best_m2_shift != -1) {
                    total_setup_cost -= delta_rem;
                    total_setup_cost += best_delta_add_shift;
                    machine_loads[m1] -= time_relief_m1;
                    machine_loads[best_m2_shift] += best_time_add_shift;

                    machine_seqs[m1].erase(machine_seqs[m1].begin() + pos1);
                    machine_seqs[best_m2_shift].insert(machine_seqs[best_m2_shift].begin() + best_pos2_shift, p1);
                    improved = true;
                    break;
                }
                
                // Move 3: INTER-MACHINE SWAP (Exchange p1 with p2 on another machine)
                int best_p2_swap = -1, best_m2_swap = -1, best_pos2_swap = -1;
                double best_net_saving_swap = 0.0;
                double best_time_add_m1 = 0.0, best_time_add_m2 = 0.0, best_time_rem_m2 = 0.0;
                
                for (int m2 = 0; m2 < nm; ++m2) {
                    if (m2 == m1) continue;
                    int sl2 = machine_seqs[m2].size();
                    for (int p2_pos = 0; p2_pos < sl2; ++p2_pos) {
                        int p2 = machine_seqs[m2][p2_pos];
                        double rate1 = data.production_rates[p1 * nm + m2];
                        double rate2 = data.production_rates[p2 * nm + m1];
                        if (rate1 <= 1e-6 || rate2 <= 1e-6) continue;

                        int prev2 = (p2_pos == 0) ? data.initial_state[m2] : machine_seqs[m2][p2_pos - 1];
                        int next2 = (p2_pos < sl2 - 1) ? machine_seqs[m2][p2_pos + 1] : -1;

                        double cost_bp2 = (next2 != -1) ? data.setup_costs[idx(prev2, next2, m2)] : 0.0;
                        double delta_rem2 = data.setup_costs[idx(prev2, p2, m2)] + ((next2 != -1) ? data.setup_costs[idx(p2, next2, m2)] : 0.0) - cost_bp2;

                        double pt2_m1 = data.demands[p2] / rate2;
                        double pt1_m2 = data.demands[p1] / rate1;
                        double pt2_m2 = data.demands[p2] / data.production_rates[p2 * nm + m2]; // FIX: Process time of P2 on M2
                        
                        double t_bp2 = (next2 != -1) ? data.setup_times[idx(prev2, next2, m2)] : 0.0;
                        double time_relief_m2 = data.setup_times[idx(prev2, p2, m2)] + pt2_m2 + ((next2 != -1) ? data.setup_times[idx(p2, next2, m2)] : 0.0) - t_bp2; // Fixed

                        // Position Swap Evaluation
                        double cost_add1 = data.setup_costs[idx(prev2, p1, m2)] + ((next2 != -1) ? data.setup_costs[idx(p1, next2, m2)] : 0.0) - cost_bp2;
                        double cost_add2 = data.setup_costs[idx(prev1, p2, m1)] + ((next1 != -1) ? data.setup_costs[idx(p2, next1, m1)] : 0.0) - cost_bp1;
                        
                        double net_saving = (delta_rem - cost_add1) + (delta_rem2 - cost_add2);
                        
                        if (net_saving > 1e-4) {
                            double time_add1 = data.setup_times[idx(prev2, p1, m2)] + ((next2 != -1) ? data.setup_times[idx(p1, next2, m2)] : 0.0) + pt1_m2 - t_bp2;
                            double time_add2 = data.setup_times[idx(prev1, p2, m1)] + ((next1 != -1) ? data.setup_times[idx(p2, next1, m1)] : 0.0) + pt2_m1 - t_bp1;
                            
                            double m1_new = machine_loads[m1] - time_relief_m1 + time_add2;
                            double m2_new = machine_loads[m2] - time_relief_m2 + time_add1;
                            
                            if (m1_new <= data.machine_capacities[m1] && m2_new <= data.machine_capacities[m2]) {
                                if (net_saving > best_net_saving_swap) {
                                    best_net_saving_swap = net_saving;
                                    best_p2_swap = p2; best_m2_swap = m2; best_pos2_swap = p2_pos;
                                    best_time_add_m1 = time_add2; best_time_add_m2 = time_add1;
                                    best_time_rem_m2 = time_relief_m2;
                                }
                            }
                        }
                    }
                }
                
                if (best_p2_swap != -1) {
                    total_setup_cost -= best_net_saving_swap;
                    machine_loads[m1] = machine_loads[m1] - time_relief_m1 + best_time_add_m1;
                    machine_loads[best_m2_swap] = machine_loads[best_m2_swap] - best_time_rem_m2 + best_time_add_m2;

                    machine_seqs[m1][pos1] = best_p2_swap;
                    machine_seqs[best_m2_swap][best_pos2_swap] = p1;
                    improved = true;
                    break;
                }
            }
        }
    }

    // =======================================================
    // HARD RE-EVALUATION & VALIDATION (Anti-Corruption)
    // Ensures incremental deltas don't drift and capacity 
    // constraints are strictly obeyed physically.
    // =======================================================
    if (penalty == 0.0) {
        double real_cost = 0.0;
        int jobs_seen = 0;
        
        for (int m = 0; m < nm; ++m) {
            double current_load = 0.0;
            int sl = machine_seqs[m].size();
            jobs_seen += sl;

            for (int pos = 0; pos < sl; ++pos) {
                int p = machine_seqs[m][pos];
                int prev = (pos == 0) ? data.initial_state[m] : machine_seqs[m][pos - 1];
                
                real_cost += data.setup_costs[idx(prev, p, m)];
                
                double rate = data.production_rates[p * nm + m];
                double pt = (rate > 1e-6) ? (data.demands[p] / rate) : 1e9;
                current_load += data.setup_times[idx(prev, p, m)] + pt;
            }
            
            if (current_load > data.machine_capacities[m] + 1e-5) {
                penalty += 1e9 + (current_load - data.machine_capacities[m]) * 1000.0; // Capacity violated physically
            }
        }
        
        if (jobs_seen != np) {
            penalty += 1e9; // Jobs were lost or duplicated
        }

        total_setup_cost = real_cost;
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

// PrintSolution for verification
void PrintSolution(TSol &s, const TProblemData &data)
{
    // Re-run logic to reconstruct (requires duplicated code or refactoring, 
    // but for RKO usually we just need OFV. This function is called at end of run)
    
    // For brevity, just print a header. The full reconstruction is same as Decoder essentially.
    printf("\n[DynamicWindowGCI Alpha=%.4f]\n", s.rk[data.num_products]);
    
    // Minimal reconstruction for logging
    const int np = data.num_products;
    const int nm = data.num_machines;
    double alpha = s.rk[np];
    
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
        // PrintSolution should always use the full logic for stable reconstruction
        bool panic_mode = false;

        int remaining = np - allocated_count;
        int k = std::max(1, (int)std::ceil(alpha * remaining));
        
        if (panic_mode) k = 1;
        // else { 
        //    const int MAX_WINDOW = 50; 
        //    if (k > MAX_WINDOW) k = MAX_WINDOW; 
        // }
        std::vector<int> candidates;
        candidates.reserve(k);
        for (int p : sorted_products) {
            if (!is_allocated[p]) {
                candidates.push_back(p);
                if ((int)candidates.size() == k) break;
            }
        }

        int best_p = -1, best_m = -1, best_pos = -1;
        double best_cost = std::numeric_limits<double>::infinity();
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
                    
                    double t_add = data.setup_times[idx(prev, p, m)];
                    double t_rem = (next!=-1)? data.setup_times[idx(prev,next,m)] : 0;
                    if(next!=-1) t_add += data.setup_times[idx(p, next, m)];
                    double d_time = t_add + prod_time - t_rem;

                    if (machine_loads[m] + d_time <= data.machine_capacities[m]) {
                        if (d_cost < best_cost) {
                            best_cost = d_cost;
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
            allocated_count++;
        } else {
            // Panic
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
