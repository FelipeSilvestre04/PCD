#ifndef _PROBLEM_SETUPGCI_H
#define _PROBLEM_SETUPGCI_H

// ============================================================================
// DECODER: DYNAMIC WINDOW SETUP COST (Hybrid GCI + SAP-SL Appending)
//
// Chromosome: size N + 2
// Genes 0..N-1: Product Priorities
// Gene N: Window Factor Alpha (0.0 <= Alpha <= 1.0) -> Size K
// Gene N+1: Strategy Gamma (Gamma > 0.5: GCI | <= 0.5: Appending)
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
    int n; // Chromosome size (N + 2)
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
    data.n = data.num_products + 2; // Needs 2 extra genes for Alpha and Gamma

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
    
    double alpha = s.rk[np];     // Gene N: Window size
    double gamma = s.rk[np+1];   // Gene N+1: Strategy selector
    
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
        int remaining = np - allocated_count;
        int k = std::max(1, (int)std::ceil(alpha * remaining));
        
        bool panic_mode = false;
        if (omp_get_wtime() - data.start_time >= data.max_time) {
            panic_mode = true;
            k = 1;
        }

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
        double best_cost = std::numeric_limits<double>::infinity();
        double best_time = 0.0;

        if (gamma > 0.5) {
            // ==========================================
            // STRATEGY A: INSERTION (GCI)
            // ==========================================
            for (int p : candidates) {
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
                        double time_add = data.setup_times[idx(prev, p, m)];
                        double time_rem = 0.0;

                        if (next != -1) {
                            cost_add += data.setup_costs[idx(p, next, m)];
                            cost_rem = data.setup_costs[idx(prev, next, m)];
                            time_add += data.setup_times[idx(p, next, m)];
                            time_rem = data.setup_times[idx(prev, next, m)];
                        }
                        
                        double delta_cost = cost_add - cost_rem;
                        double delta_time = time_add + prod_time - time_rem;

                        if (machine_loads[m] + delta_time <= data.machine_capacities[m]) {
                            if (delta_cost < best_cost) {
                                best_cost = delta_cost;
                                best_m = m;
                                best_pos = pos;
                                best_time = delta_time;
                                best_p = p;
                            }
                        }
                    }
                }
            }
        } else {
            // ==========================================
            // STRATEGY B: APPENDING (SAP-SL inspired)
            // ==========================================
            for (int p : candidates) {
                for (int m = 0; m < nm; ++m) {
                    double rate = data.production_rates[p * nm + m];
                    if (rate <= 1e-6) continue;

                    double prod_time = data.demands[p] / rate;
                    const auto& seq = machine_seqs[m];
                    int pos = seq.size(); // Append only at the end
                    
                    int prev = (pos == 0) ? data.initial_state[m] : seq[pos - 1];
                    
                    double delta_cost = data.setup_costs[idx(prev, p, m)];
                    double delta_time = data.setup_times[idx(prev, p, m)] + prod_time;

                    if (machine_loads[m] + delta_time <= data.machine_capacities[m]) {
                        if (delta_cost < best_cost) {
                            best_cost = delta_cost;
                            best_m = m;
                            best_pos = pos;
                            best_time = delta_time;
                            best_p = p;
                        }
                    }
                }
            }
        }

        // =======================================================
        // ALLOTMENT OR FALLBACK COMPUTATION
        // =======================================================
        if (best_p != -1) {
            machine_seqs[best_m].insert(machine_seqs[best_m].begin() + best_pos, best_p);
            machine_loads[best_m] += best_time;
            total_setup_cost += best_cost;
            is_allocated[best_p] = true;
            allocated_count++;
        } else {
            // FALLBACK 1: Try ALL remaining products to find ANY valid insertion
            best_cost = std::numeric_limits<double>::infinity();
            
            for (int p : sorted_products) {
                if (is_allocated[p]) continue;
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
                        double time_add = data.setup_times[idx(prev, p, m)];
                        double time_rem = 0.0;

                        if (next != -1) {
                            cost_add += data.setup_costs[idx(p, next, m)];
                            cost_rem = data.setup_costs[idx(prev, next, m)];
                            time_add += data.setup_times[idx(p, next, m)];
                            time_rem = data.setup_times[idx(prev, next, m)];
                        }
                        
                        double delta_time = time_add + prod_time - time_rem;

                        if (machine_loads[m] + delta_time <= data.machine_capacities[m]) {
                            double delta_cost = cost_add - cost_rem;
                            if (delta_cost < best_cost) {
                                best_cost = delta_cost;
                                best_m = m;
                                best_pos = pos;
                                best_time = delta_time;
                                best_p = p;
                            }
                        }
                    }
                }
            }

            if (best_p != -1) {
                machine_seqs[best_m].insert(machine_seqs[best_m].begin() + best_pos, best_p);
                machine_loads[best_m] += best_time;
                total_setup_cost += best_cost;
                is_allocated[best_p] = true;
                allocated_count++;
            } else {
                // FALLBACK 2: HARD PENALTY (Force Append in the most empty machine) To prevent infinite loops
                int panic_p = candidates[0]; // Highest priority unallocated in this window
                
                best_m = 0;
                double max_rem_cap = data.machine_capacities[0] - machine_loads[0];
                for(int m=1; m<nm; ++m) {
                    double rem_cap = data.machine_capacities[m] - machine_loads[m];
                    if(rem_cap > max_rem_cap) {
                        max_rem_cap = rem_cap;
                        best_m = m;
                    }
                }
                
                int pos = machine_seqs[best_m].size();
                int prev = (pos == 0) ? data.initial_state[best_m] : machine_seqs[best_m].back();
                double delta_cost = data.setup_costs[idx(prev, panic_p, best_m)];
                
                machine_seqs[best_m].push_back(panic_p);
                
                double rate = data.production_rates[panic_p * nm + best_m];
                double prod_time = (rate > 1e-6) ? (data.demands[panic_p] / rate) : 0.0;
                double time_add = data.setup_times[idx(prev, panic_p, best_m)];
                
                machine_loads[best_m] += (time_add + prod_time);
                total_setup_cost += delta_cost;
                
                penalty += 1e9 + data.demands[panic_p] * 1000.0;
                is_allocated[panic_p] = true;
                allocated_count++;
            }
        }
    }

    // =======================================================
    // LOCAL SEARCH: IMI with Speed-Up (Worst-Edge focused)
    // Ref: Vallada and Ruiz (2011) Speed-up procedure
    // =======================================================
    if (penalty == 0.0) { // Only refine feasible solutions
        int ls_iters = 0;
        bool improved = true;
        while (improved && ls_iters < 40) { // Increased iterations
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
            int limit = std::min(15, (int)edges.size());
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

                // Move 1: SHIFT (Insertion into other machines)
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
                
                // Move 2: SWAP (Exchange p1 with p2 on another machine)
                // This breaks full-capacity blocks where shifts alone fail!
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
                        double t_bp2 = (next2 != -1) ? data.setup_times[idx(prev2, next2, m2)] : 0.0;
                        double time_relief_m2 = data.setup_times[idx(prev2, p2, m2)] + pt2_m1 + ((next2 != -1) ? data.setup_times[idx(p2, next2, m2)] : 0.0) - t_bp2;

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
    double alpha = s.rk[np];
    double gamma = s.rk[np+1];
    printf("\n[SetupGCI Alpha=%.4f | Gamma=%.4f]\n", alpha, gamma);
    
    // Quick rebuild to print 
    const int nm = data.num_machines;
    std::vector<int> sorted_products(np);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) { return s.rk[i] < s.rk[j]; });

    std::vector<bool> is_allocated(np, false);
    int allocated_count = 0;
    std::vector<std::vector<int>> machine_seqs(nm);
    std::vector<double> machine_loads(nm, 0.0);

    auto idx = [np, nm](int prev, int curr, int m) { return (prev * np * nm) + (curr * nm) + m; };

    while (allocated_count < np) {
        int rem = np - allocated_count;
        int k = std::max(1, (int)std::ceil(alpha * rem));
        std::vector<int> cands;
        for(int p: sorted_products){if(!is_allocated[p]){cands.push_back(p);if((int)cands.size()==k)break;}}
        
        int best_p=-1, best_m=-1, best_pos=-1; double best_cost=1e18, best_time=0;
        
        if (gamma > 0.5) {
            for(int p:cands)for(int m=0;m<nm;++m){
                double r=data.production_rates[p*nm+m]; if(r<=1e-6)continue;
                double pt=data.demands[p]/r; int sl=machine_seqs[m].size();
                for(int pos=0;pos<=sl;++pos){
                    int pv=(pos==0)?data.initial_state[m]:machine_seqs[m][pos-1];
                    int nx=(pos<sl)?machine_seqs[m][pos]:-1;
                    double ca=data.setup_costs[idx(pv,p,m)], cr=0, ta=data.setup_times[idx(pv,p,m)], tr=0;
                    if(nx!=-1){ca+=data.setup_costs[idx(p,nx,m)];cr=data.setup_costs[idx(pv,nx,m)];ta+=data.setup_times[idx(p,nx,m)];tr=data.setup_times[idx(pv,nx,m)];}
                    double dt=ta+pt-tr;
                    if(machine_loads[m]+dt<=data.machine_capacities[m]){
                        double dc=ca-cr; if(dc<best_cost){best_cost=dc;best_m=m;best_pos=pos;best_time=dt;best_p=p;}
                    }
                }
            }
        } else {
            for(int p:cands)for(int m=0;m<nm;++m){
                double r=data.production_rates[p*nm+m]; if(r<=1e-6)continue;
                double pt=data.demands[p]/r; int pos=machine_seqs[m].size();
                int pv=(pos==0)?data.initial_state[m]:machine_seqs[m][pos-1];
                double dc=data.setup_costs[idx(pv,p,m)], dt=data.setup_times[idx(pv,p,m)]+pt;
                if(machine_loads[m]+dt<=data.machine_capacities[m]){
                    if(dc<best_cost){best_cost=dc;best_m=m;best_pos=pos;best_time=dt;best_p=p;}
                }
            }
        }
        
        if(best_p==-1){
            for(int p:sorted_products){if(is_allocated[p])continue;
                for(int m=0;m<nm;++m){double r=data.production_rates[p*nm+m];if(r<=1e-6)continue;
                    double pt=data.demands[p]/r;int sl=machine_seqs[m].size();
                    for(int pos=0;pos<=sl;++pos){
                        int pv=(pos==0)?data.initial_state[m]:machine_seqs[m][pos-1];
                        int nx=(pos<sl)?machine_seqs[m][pos]:-1;
                        double ca=data.setup_costs[idx(pv,p,m)], cr=0, ta=data.setup_times[idx(pv,p,m)], tr=0;
                        if(nx!=-1){ca+=data.setup_costs[idx(p,nx,m)];cr=data.setup_costs[idx(pv,nx,m)];ta+=data.setup_times[idx(p,nx,m)];tr=data.setup_times[idx(pv,nx,m)];}
                        double dt=ta+pt-tr;
                        if(machine_loads[m]+dt<=data.machine_capacities[m]){
                            double dc=ca-cr; if(dc<best_cost){best_cost=dc;best_m=m;best_pos=pos;best_time=dt;best_p=p;}
                        }}}}
        }

        if(best_p!=-1){machine_seqs[best_m].insert(machine_seqs[best_m].begin()+best_pos,best_p);machine_loads[best_m]+=best_time;is_allocated[best_p]=true;allocated_count++;}
        else{int panic=cands[0]; best_m=0;double mc=data.machine_capacities[0]-machine_loads[0];for(int m=1;m<nm;++m){double c=data.machine_capacities[m]-machine_loads[m];if(c>mc){mc=c;best_m=m;}}
             machine_seqs[best_m].push_back(panic);is_allocated[panic]=true;allocated_count++;}
    }

    printf("=== SEQUENCE_START ===\n[");
    for (int m = 0; m < nm; ++m) {
        printf("[");
        for (size_t i = 0; i < machine_seqs[m].size(); ++i) {
            printf("%d", machine_seqs[m][i]);
            if (i < machine_seqs[m].size() - 1) printf(", ");
        }
        printf("]");
        if (m < nm - 1) printf(", ");
    }
    printf("]\n=== SEQUENCE_END ===\n");
}

#endif
