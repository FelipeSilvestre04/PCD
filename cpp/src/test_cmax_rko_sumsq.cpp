#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <random>
#include <omp.h>
#include <fstream>
#include <sstream>

struct TSol {
    std::vector<double> rk;
    double ofv;
    char nameMH[100];
};

#include "Problem/Problem_DynamicWindowGCI_Cmax_SumSquares.h"

// Helper to get sequences as string for CSV
std::string GetSequenceString(TSol &s, const TProblemData &data) {
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

    std::stringstream ss;
    ss << "[";
    for (int m = 0; m < M; ++m) {
        ss << "[";
        for (size_t i = 0; i < machine_seqs[m].size(); ++i) {
            ss << machine_seqs[m][i];
            if (i < machine_seqs[m].size() - 1) ss << ", ";
        }
        ss << "]";
        if (m < M - 1) ss << ", ";
    }
    ss << "]";
    return ss.str();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <instance_path> <time_limit_s> <bks>" << std::endl;
        return 1;
    }

    char* instance_path = argv[1];
    double time_limit = std::stod(argv[2]);
    int bks = std::stoi(argv[3]);

    TProblemData data;
    ReadData(instance_path, data);
    data.start_time = omp_get_wtime();
    data.max_time = time_limit;

    TSol bestSol;
    bestSol.ofv = 1e18;
    long long iterations = 0;
    
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto t_start = std::chrono::high_resolution_clock::now();

    while (true) {
        auto t_now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_now - t_start).count();
        if (elapsed >= time_limit) break;

        TSol s;
        s.rk.resize(data.n);
        for(int i=0; i<data.n; ++i) s.rk[i] = dist(rng);

        s.ofv = Decoder(s, data);

        if (s.ofv < bestSol.ofv) {
            bestSol = s;
            double gap = (bestSol.ofv - bks) / bks * 100.0;
            printf("[%.3fs] New Best: %.0f (Gap: %.2f%%) | Iter: %lld\n", elapsed, bestSol.ofv, gap, iterations);
        }
        iterations++;
    }

    double final_gap = (bestSol.ofv - bks) / bks * 100.0;
    
    std::string inst_str(instance_path);
    size_t last_slash = inst_str.find_last_of("\\/");
    std::string inst_name = (last_slash == std::string::npos) ? inst_str : inst_str.substr(last_slash + 1);
    if (inst_name.find(".txt") != std::string::npos) inst_name = inst_name.substr(0, inst_name.find(".txt"));

    // Save results
    std::ofstream f_res("results_sumsq.csv", std::ios::app);
    if (f_res.is_open()) {
        f_res << inst_name << ";" << bestSol.ofv << ";" << final_gap << ";" << iterations << ";" << time_limit << "\n";
        f_res.close();
    }

    // Save sequences
    std::string seq_str = GetSequenceString(bestSol, data);
    std::ofstream f_seq("sequences_sumsq.csv", std::ios::app);
    if (f_seq.is_open()) {
        f_seq << inst_name << ";" << seq_str << "\n";
        f_seq.close();
    }

    printf("\nFINAL_RESULT;%s;%.0f;%.2f;%lld;%.1f\n", instance_path, bestSol.ofv, final_gap, iterations, time_limit);
    
    FreeMemoryProblem(data);
    return 0;
}
