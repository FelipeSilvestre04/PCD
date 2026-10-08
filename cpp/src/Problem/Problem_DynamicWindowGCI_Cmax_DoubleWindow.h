#ifndef _PROBLEM_DYNAMICWINDOWGCI_CMAX_DOUBLEWINDOW_H
#define _PROBLEM_DYNAMICWINDOWGCI_CMAX_DOUBLEWINDOW_H

/**
 * DECODER: DYNAMIC WINDOW GLOBAL CHEAPEST INSERTION (Cmax / Vallada 2011)
 * with DOUBLE WINDOWING (Alpha for Jobs, Beta for Machines)
 *
 * Target: Rm | s_jk | Cmax (Unrelated Parallel Machines, Sequence-Dependent Setup)
 * Chromosome: size N + M + 2
 *   Genes [0..N-1]:     Job Priorities
 *   Genes [N..N+M-1]:   Machine Priorities
 *   Gene  [N+M]:        Alpha (Job window factor)
 *   Gene  [N+M+1]:      Beta  (Machine window factor)
 */

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>

struct TProblemData {
    int n; // Chromosome size (N + M + 2)
    int num_jobs;
    int num_machines;
    
    // Vallada Format
    std::vector<std::vector<int>> processing; // [m][j]
    std::vector<std::vector<std::vector<int>>> setup; // [m][i][j]

    double start_time = 0.0;
    double max_time = 1e9;
};

void ReadData(char name[], TProblemData &data) {
    std::ifstream file(name);
    if (!file.is_open()) {
        printf("\nERROR: File (%s) not found!\n", name);
        exit(1);
    }

    int dummy_num, num_check;
    file >> data.num_jobs >> data.num_machines >> dummy_num;
    file >> num_check;

    int N = data.num_jobs;
    int M = data.num_machines;
    data.n = N + M + 2;

    data.processing.assign(M, std::vector<int>(N + 1, 0));
    for (int j = 1; j <= N; ++j) {
        for (int k = 0; k < M; ++k) {
            int mid, ptime;
            file >> mid >> ptime;
            data.processing[mid][j] = ptime;
        }
    }

    std::string token;
    while (file >> token && token != "SSD");

    data.setup.assign(M, std::vector<std::vector<int>>(N + 1, std::vector<int>(N + 1, 0)));
    for (int m = 0; m < M; ++m) {
        file >> token; // M0, M1...
        for (int i = 1; i <= N; ++i) {
            for (int j = 1; j <= N; ++j) {
                file >> data.setup[m][i][j];
            }
        }
        // Initial setup convention: s[0][j] = s[j][j]
        for (int j = 1; j <= N; ++j) {
            data.setup[m][0][j] = data.setup[m][j][j];
        }
    }
    file.close();
}

double Decoder(TSol &s, const TProblemData &data) {
    const int N = data.num_jobs;
    const int M = data.num_machines;

    // Extract alpha and beta from chromosome
    double alpha = s.rk[N + M];       // Gene [N+M]
    double beta  = s.rk[N + M + 1];   // Gene [N+M+1]

    std::vector<int> sorted_jobs(N);
    std::iota(sorted_jobs.begin(), sorted_jobs.end(), 1);
    std::stable_sort(sorted_jobs.begin(), sorted_jobs.end(), [&](int i, int j) {
        if (s.rk[i-1] != s.rk[j-1])
            return s.rk[i-1] < s.rk[j-1];
        return i < j;
    });

    // Sort machines by keys [N..N+M-1]
    std::vector<int> sorted_machines(M);
    std::iota(sorted_machines.begin(), sorted_machines.end(), 0);
    std::stable_sort(sorted_machines.begin(), sorted_machines.end(), [&](int a, int b) {
        if (s.rk[N + a] != s.rk[N + b])
            return s.rk[N + a] < s.rk[N + b];
        return a < b;
    });

    // Machine window size
    int k_maq = std::max(1, (int)std::ceil(beta * M));
    std::vector<int> machine_window(sorted_machines.begin(), sorted_machines.begin() + k_maq);
    
    // Build full machine list for fallback
    std::vector<int> all_machines(M);
    std::iota(all_machines.begin(), all_machines.end(), 0);

    std::vector<bool> is_allocated(N + 1, false);
    int allocated_count = 0;

    std::vector<std::vector<int>> machine_seqs(M);
    std::vector<int> machine_loads(M, 0);
    int current_makespan = 0;

    while (allocated_count < N) {
        int remaining = N - allocated_count;
        int k_prod = std::max(1, (int)std::ceil(alpha * remaining));

        std::vector<int> candidates;
        for (int j : sorted_jobs) {
            if (!is_allocated[j]) {
                candidates.push_back(j);
                if ((int)candidates.size() == k_prod) break;
            }
        }

        int best_j = -1, best_m = -1, best_pos = -1;
        int best_new_makespan = std::numeric_limits<int>::max();
        int best_delta = std::numeric_limits<int>::max();
        int best_setup_sum = std::numeric_limits<int>::max();
        int best_setup_arcs = 1;

        // Helper function for insertion check
        auto try_insertion = [&](const std::vector<int>& machines) {
            for (int j : candidates) {
                for (int m : machines) {
                    const auto& seq = machine_seqs[m];
                    int seq_len = seq.size();
#ifdef MM_CMAX_TAIL_GCI
                    const int first_pos = seq_len;
#else
                    const int first_pos = 0;
#endif
                    for (int pos = first_pos; pos <= seq_len; ++pos) {
                        int prev = (pos == 0) ? 0 : seq[pos - 1];
                        int next = (pos < seq_len) ? seq[pos] : -1;

                        int delta = data.setup[m][prev][j] + data.processing[m][j];
                        if (next != -1) delta += data.setup[m][j][next] - data.setup[m][prev][next];

                        const int setup_sum = data.setup[m][prev][j]
                            + ((next != -1) ? data.setup[m][j][next] : 0);
                        const int setup_arcs = (next != -1) ? 2 : 1;

                        int new_m_load = machine_loads[m] + delta;
                        int new_makespan = new_m_load;
                        for (int other = 0; other < M; ++other) {
                            if (other != m) new_makespan = std::max(new_makespan, machine_loads[other]);
                        }

                        const bool lower_setup_average =
                            (long long)setup_sum * best_setup_arcs
                            < (long long)best_setup_sum * setup_arcs;
                        const bool equal_setup_average =
                            (long long)setup_sum * best_setup_arcs
                            == (long long)best_setup_sum * setup_arcs;
                        if (new_makespan < best_new_makespan ||
                            (new_makespan == best_new_makespan &&
                             (lower_setup_average ||
                              (equal_setup_average && delta < best_delta)))) {
                            best_new_makespan = new_makespan;
                            best_delta = delta;
                            best_setup_sum = setup_sum;
                            best_setup_arcs = setup_arcs;
                            best_j = j;
                            best_m = m;
                            best_pos = pos;
                        }
                    }
                }
            }
        };

        // 1. Try with machine window
        try_insertion(machine_window);

        // 2. FALLBACK: If window is too restrictive, we could retry with all, 
        // but for Makespan, every machine is technically feasible. 
        // V2.h logic says fallback if no "feasible" insertion. 
        // Here, we'll only fallback if best_j == -1 (unlikely in pure Cmax).
        if (best_j == -1) {
            try_insertion(all_machines);
        }

        if (best_j != -1) {
            machine_seqs[best_m].insert(machine_seqs[best_m].begin() + best_pos, best_j);
            machine_loads[best_m] += best_delta;
            current_makespan = *std::max_element(machine_loads.begin(), machine_loads.end());
            is_allocated[best_j] = true;
            allocated_count++;
        } else {
             printf("\nERROR: No valid insertion found for candidates!\n");
             exit(1);
        }
    }

    long long total_load = 0;
    for (int m = 0; m < M; ++m) {
        total_load += machine_loads[m];
    }

    return (double)current_makespan + (total_load * 1e-6);
}

void PrintSolution(TSol &s, const TProblemData &data) {
    const int N = data.num_jobs;
    const int M = data.num_machines;
    
    double alpha = s.rk[N + M];
    double beta  = s.rk[N + M + 1];

    printf("\n[Cmax DoubleWindow Alpha=%.4f Beta=%.4f]\n", alpha, beta);

    // Re-run decoder logic to get final state
    std::vector<int> sorted_jobs(N);
    std::iota(sorted_jobs.begin(), sorted_jobs.end(), 1);
    std::stable_sort(sorted_jobs.begin(), sorted_jobs.end(), [&](int i, int j) {
        if (s.rk[i-1] != s.rk[j-1])
            return s.rk[i-1] < s.rk[j-1];
        return i < j;
    });

    std::vector<int> sorted_machines(M);
    std::iota(sorted_machines.begin(), sorted_machines.end(), 0);
    std::stable_sort(sorted_machines.begin(), sorted_machines.end(), [&](int a, int b) {
        if (s.rk[N + a] != s.rk[N + b])
            return s.rk[N + a] < s.rk[N + b];
        return a < b;
    });

    int k_maq = std::max(1, (int)std::ceil(beta * M));
    std::vector<int> machine_window(sorted_machines.begin(), sorted_machines.begin() + k_maq);
    std::vector<int> all_machines(M);
    std::iota(all_machines.begin(), all_machines.end(), 0);

    std::vector<bool> is_allocated(N + 1, false);
    int allocated_count = 0;
    std::vector<std::vector<int>> machine_seqs(M);
    std::vector<int> machine_loads(M, 0);
    int current_makespan = 0;

    while (allocated_count < N) {
        int remaining = N - allocated_count;
        int k_prod = std::max(1, (int)std::ceil(alpha * remaining));
        std::vector<int> candidates;
        for (int j : sorted_jobs) {
            if (!is_allocated[j]) {
                candidates.push_back(j);
                if ((int)candidates.size() == k_prod) break;
            }
        }

        int best_j = -1, best_m = -1, best_pos = -1;
        int best_new_makespan = std::numeric_limits<int>::max();
        int best_delta = std::numeric_limits<int>::max();
        int best_setup_sum = std::numeric_limits<int>::max();
        int best_setup_arcs = 1;

        auto try_insert_print = [&](const std::vector<int>& machines) {
            for (int j : candidates) {
                for (int m : machines) {
                    const auto& seq = machine_seqs[m];
                    int seq_len = seq.size();
#ifdef MM_CMAX_TAIL_GCI
                    const int first_pos = seq_len;
#else
                    const int first_pos = 0;
#endif
                    for (int pos = first_pos; pos <= seq_len; ++pos) {
                        int prev = (pos == 0) ? 0 : seq[pos - 1];
                        int next = (pos < seq_len) ? seq[pos] : -1;
                        int delta = data.setup[m][prev][j] + data.processing[m][j];
                        if (next != -1) delta += data.setup[m][j][next] - data.setup[m][prev][next];
                        const int setup_sum = data.setup[m][prev][j]
                            + ((next != -1) ? data.setup[m][j][next] : 0);
                        const int setup_arcs = (next != -1) ? 2 : 1;
                        int new_m_load = machine_loads[m] + delta;
                        int new_makespan = new_m_load;
                        for (int other = 0; other < M; ++other) {
                            if (other != m) new_makespan = std::max(new_makespan, machine_loads[other]);
                        }
                        const bool lower_setup_average =
                            (long long)setup_sum * best_setup_arcs
                            < (long long)best_setup_sum * setup_arcs;
                        const bool equal_setup_average =
                            (long long)setup_sum * best_setup_arcs
                            == (long long)best_setup_sum * setup_arcs;
                        if (new_makespan < best_new_makespan ||
                            (new_makespan == best_new_makespan &&
                             (lower_setup_average ||
                              (equal_setup_average && delta < best_delta)))) {
                            best_new_makespan = new_makespan; best_delta = delta;
                            best_setup_sum = setup_sum; best_setup_arcs = setup_arcs;
                            best_j = j; best_m = m; best_pos = pos;
                        }
                    }
                }
            }
        };

        try_insert_print(machine_window);
        if (best_j == -1) try_insert_print(all_machines);

        if (best_j != -1) {
            machine_seqs[best_m].insert(machine_seqs[best_m].begin() + best_pos, best_j);
            machine_loads[best_m] += best_delta;
            current_makespan = *std::max_element(machine_loads.begin(), machine_loads.end());
            is_allocated[best_j] = true;
            allocated_count++;
        } else {
            break;
        }
    }

    printf("[Actual Makespan: %d]\n", current_makespan);
    printf("[");
    for (int m = 0; m < M; ++m) {
        printf("[");
        for (size_t i = 0; i < machine_seqs[m].size(); ++i) {
            printf("%d", machine_seqs[m][i]);
            if (i < machine_seqs[m].size() - 1) printf(", ");
        }
        printf("]");
        if (m < M - 1) printf(", ");
    }
    printf("]\n");
}

void FreeMemoryProblem(TProblemData &data) {
    data.processing.clear();
    data.setup.clear();
}

#endif
