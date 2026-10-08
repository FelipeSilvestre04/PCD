#ifndef _PROBLEM_DYNAMICWINDOWGCI_CMAX_SUMSQUARES_H
#define _PROBLEM_DYNAMICWINDOWGCI_CMAX_SUMSQUARES_H

/**
 * DECODER: DYNAMIC WINDOW GLOBAL CHEAPEST INSERTION (Cmax + Sum of Squares Tie-breaker)
 *
 * Target: Rm | s_jk | Cmax (Unrelated Parallel Machines, Sequence-Dependent Setup)
 * This version uses the Sum of Squares of machine loads as a tie-breaker to promote load balancing.
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
    int n; // Chromosome size (N + 1)
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
    data.n = N + 1;

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
        for (int j = 1; j <= N; ++j) {
            data.setup[m][0][j] = data.setup[m][j][j];
        }
    }
    file.close();
}

double Decoder(TSol &s, const TProblemData &data) {
    const int N = data.num_jobs;
    const int M = data.num_machines;

    double alpha = s.rk[N];

    std::vector<int> sorted_jobs(N);
    std::iota(sorted_jobs.begin(), sorted_jobs.end(), 1);
    std::stable_sort(sorted_jobs.begin(), sorted_jobs.end(), [&](int i, int j) {
        if (std::abs(s.rk[i-1] - s.rk[j-1]) > 1e-9)
            return s.rk[i-1] < s.rk[j-1];
        return i < j;
    });

    std::vector<bool> is_allocated(N + 1, false);
    int allocated_count = 0;

    std::vector<std::vector<int>> machine_seqs(M);
    std::vector<int> machine_loads(M, 0);
    int current_makespan = 0;

    while (allocated_count < N) {
        int remaining = N - allocated_count;
        int k = std::max(1, (int)std::ceil(alpha * remaining));

        std::vector<int> candidates;
        for (int j : sorted_jobs) {
            if (!is_allocated[j]) {
                candidates.push_back(j);
                if ((int)candidates.size() == k) break;
            }
        }

        int best_j = -1, best_m = -1, best_pos = -1;
        int best_new_makespan = std::numeric_limits<int>::max();
        long long best_new_ssq = std::numeric_limits<long long>::max();

        // Calculate current Sum of Squares
        long long current_ssq = 0;
        for (int m = 0; m < M; ++m) {
            current_ssq += (long long)machine_loads[m] * machine_loads[m];
        }

        for (int j : candidates) {
            for (int m = 0; m < M; ++m) {
                const auto& seq = machine_seqs[m];
                int seq_len = seq.size();
                for (int pos = 0; pos <= seq_len; ++pos) {
                    int prev = (pos == 0) ? 0 : seq[pos - 1];
                    int next = (pos < seq_len) ? seq[pos] : -1;

                    int delta = data.setup[m][prev][j] + data.processing[m][j];
                    if (next != -1) delta += data.setup[m][j][next] - data.setup[m][prev][next];

                    int new_m_load = machine_loads[m] + delta;
                    int new_makespan = std::max(new_m_load, current_makespan);
                    
                    // Sum of Squares update: S' = S - Lm^2 + (Lm + delta)^2
                    long long new_ssq = current_ssq - ((long long)machine_loads[m] * machine_loads[m]) 
                                                   + ((long long)new_m_load * new_m_load);

                    if (new_makespan < best_new_makespan || (new_makespan == best_new_makespan && new_ssq < best_new_ssq)) {
                        best_new_makespan = new_makespan;
                        best_new_ssq = new_ssq;
                        best_j = j;
                        best_m = m;
                        best_pos = pos;
                    }
                }
            }
        }

        if (best_j != -1) {
            machine_seqs[best_m].insert(machine_seqs[best_m].begin() + best_pos, best_j);
            machine_loads[best_m] += (best_new_ssq == std::numeric_limits<long long>::max() ? 0 : 0); // placeholder
            
            // Re-calculating delta from chosen parameters
            int prev = (best_pos == 0) ? 0 : machine_seqs[best_m][best_pos - 1];
            int next = (best_pos + 1 < (int)machine_seqs[best_m].size()) ? machine_seqs[best_m][best_pos + 1] : -1;
            int delta = data.setup[best_m][prev][best_j] + data.processing[best_m][best_j];
            if (next != -1) delta += data.setup[best_m][best_j][next] - data.setup[best_m][prev][next];
            
            machine_loads[best_m] += delta;
            current_makespan = best_new_makespan;
            is_allocated[best_j] = true;
            allocated_count++;
        } else {
             int pj = candidates[0];
             is_allocated[pj] = true;
             allocated_count++;
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
    double alpha = s.rk[N];
    
    std::vector<int> sorted_jobs(N);
    std::iota(sorted_jobs.begin(), sorted_jobs.end(), 1);
    std::stable_sort(sorted_jobs.begin(), sorted_jobs.end(), [&](int i, int j) {
        if (std::abs(s.rk[i-1] - s.rk[j-1]) > 1e-9)
            return s.rk[i-1] < s.rk[j-1];
        return i < j;
    });

    std::vector<bool> is_allocated(N + 1, false);
    int allocated_count = 0;
    std::vector<std::vector<int>> machine_seqs(M);
    std::vector<int> machine_loads(M, 0);
    int current_makespan = 0;

    while (allocated_count < N) {
        int remaining = N - allocated_count;
        int k = std::max(1, (int)std::ceil(alpha * remaining));
        std::vector<int> candidates;
        for (int j : sorted_jobs) {
            if (!is_allocated[j]) {
                candidates.push_back(j);
                if ((int)candidates.size() == k) break;
            }
        }

        int best_j = -1, best_m = -1, best_pos = -1;
        int best_new_makespan = std::numeric_limits<int>::max();
        long long best_new_ssq = std::numeric_limits<long long>::max();

        long long current_ssq = 0;
        for (int m = 0; m < M; ++m) current_ssq += (long long)machine_loads[m] * machine_loads[m];

        for (int j : candidates) {
            for (int m = 0; m < M; ++m) {
                const auto& seq = machine_seqs[m];
                int seq_len = seq.size();
                for (int pos = 0; pos <= seq_len; ++pos) {
                    int prev = (pos == 0) ? 0 : seq[pos - 1];
                    int next = (pos < seq_len) ? seq[pos] : -1;
                    int delta = data.setup[m][prev][j] + data.processing[m][j];
                    if (next != -1) delta += data.setup[m][j][next] - data.setup[m][prev][next];
                    int new_m_load = machine_loads[m] + delta;
                    int new_makespan = std::max(new_m_load, current_makespan);
                    long long new_ssq = current_ssq - ((long long)machine_loads[m] * machine_loads[m]) + ((long long)new_m_load * new_m_load);
                    if (new_makespan < best_new_makespan || (new_makespan == best_new_makespan && new_ssq < best_new_ssq)) {
                        best_new_makespan = new_makespan;
                        best_new_ssq = new_ssq;
                        best_j = j;
                        best_m = m;
                        best_pos = pos;
                    }
                }
            }
        }
        if (best_j != -1) {
            int prev = (best_pos == 0) ? 0 : machine_seqs[best_m][best_pos - 1];
            int next = (best_pos < (int)machine_seqs[best_m].size()) ? machine_seqs[best_m][best_pos] : -1;
            int delta = data.setup[best_m][prev][best_j] + data.processing[best_m][best_j];
            if (next != -1) delta += data.setup[best_m][best_j][next] - data.setup[best_m][prev][next];

            machine_seqs[best_m].insert(machine_seqs[best_m].begin() + best_pos, best_j);
            machine_loads[best_m] += delta;
            current_makespan = best_new_makespan;
            is_allocated[best_j] = true;
            allocated_count++;
        } else break;
    }

    printf("\n[SumSquares Decoder] OFV: %d\n", current_makespan);
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
