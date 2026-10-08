#ifndef _PROBLEM_NN_H
#define _PROBLEM_NN_H

// ============================================================================
// DECODER: NEAREST NEIGHBOR / TAIL INSERTION (List Scheduling)
// Para cada produto, testa apenas a posição FINAL de cada máquina
// Muito mais rápido que Cheapest Insertion!
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
// 3. DECODER: NEAREST NEIGHBOR (Tail Insertion / List Scheduling)
// ============================================================================
// Para cada produto na ordem das keys:
//   - Testa apenas a posição FINAL de cada máquina
//   - Escolhe a máquina com menor custo de setup (do último produto para este)
//   - Adiciona ao final (append) dessa máquina
//
// Complexidade: O(n * m) vs O(n * m * n) do Cheapest Insertion
// ============================================================================

double Decoder(TSol &s, const TProblemData &data)
{
    const int np = data.num_products;
    const int nm = data.num_machines;
    
    // 1. Sort products by keys (order determined by RKO/GA)
    std::vector<int> sorted_products(np);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) {
        return s.rk[i] < s.rk[j];
    });

    // Solution structures
    std::vector<std::vector<int>> machine_seqs(nm);
    std::vector<double> machine_loads(nm, 0.0);
    
    // Track the current last product of each machine
    std::vector<int> current_last(nm);
    for (int m = 0; m < nm; ++m) {
        current_last[m] = data.initial_state[m];
    }
    
    double total_setup_cost = 0.0;
    double penalty = 0.0;

    // Index helper lambda
    auto idx = [np, nm](int prev, int curr, int m) {
        return (prev * np * nm) + (curr * nm) + m;
    };

    // 2. List Scheduling with Tail Insertion (Nearest Neighbor style)
    for (int p : sorted_products) {
        int best_machine = -1;
        double min_cost = std::numeric_limits<double>::infinity();
        double best_time = 0.0;

        for (int m = 0; m < nm; ++m) {
            double rate = data.production_rates[p * nm + m];
            if (rate <= 1e-6) continue;

            double prod_time = data.demands[p] / rate;
            
            // Only test TAIL POSITION (append to end)
            int prev = current_last[m];
            
            // Cost = setup from last product to this product
            double setup_cost = data.setup_costs[idx(prev, p, m)];
            double setup_time = data.setup_times[idx(prev, p, m)];
            double total_time = setup_time + prod_time;

            // Check capacity
            if (machine_loads[m] + total_time <= data.machine_capacities[m]) {
                // Choose minimum cost (Nearest Neighbor criterion)
                if (setup_cost < min_cost) {
                    min_cost = setup_cost;
                    best_machine = m;
                    best_time = total_time;
                }
            }
        }

        // Perform allocation (append to end)
        if (best_machine != -1) {
            machine_seqs[best_machine].push_back(p);
            machine_loads[best_machine] += best_time;
            current_last[best_machine] = p;
            total_setup_cost += min_cost;
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
        return s.rk[i] < s.rk[j];
    });

    std::vector<std::vector<int>> machine_seqs(nm);
    std::vector<double> machine_loads(nm, 0.0);
    std::vector<int> current_last(nm);
    for (int m = 0; m < nm; ++m) {
        current_last[m] = data.initial_state[m];
    }

    auto idx = [np, nm](int prev, int curr, int m) {
        return (prev * np * nm) + (curr * nm) + m;
    };

    for (int p : sorted_products) {
        int best_machine = -1;
        double min_cost = std::numeric_limits<double>::infinity();
        double best_time = 0.0;

        for (int m = 0; m < nm; ++m) {
            double rate = data.production_rates[p * nm + m];
            if (rate <= 1e-6) continue;
            double prod_time = data.demands[p] / rate;
            
            int prev = current_last[m];
            double setup_cost = data.setup_costs[idx(prev, p, m)];
            double setup_time = data.setup_times[idx(prev, p, m)];
            double total_time = setup_time + prod_time;

            if (machine_loads[m] + total_time <= data.machine_capacities[m]) {
                if (setup_cost < min_cost) {
                    min_cost = setup_cost;
                    best_machine = m;
                    best_time = total_time;
                }
            }
        }
        if (best_machine != -1) {
            machine_seqs[best_machine].push_back(p);
            machine_loads[best_machine] += best_time;
            current_last[best_machine] = p;
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
