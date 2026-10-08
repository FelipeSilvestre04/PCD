#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <random>
#include <omp.h>

// Base types needed by the decoder
struct TSol {
    std::vector<double> rk;
    double ofv;
    char nameMH[100];
};

// Include the new decoder with Double Windowing
#include "Problem/Problem_DynamicWindowGCI_Cmax_DoubleWindow.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <instance_file> <time_limit_s> [bks]" << std::endl;
        return 1;
    }

    char* instance_file = argv[1];
    double time_limit = std::stod(argv[2]);
    int bks = (argc > 3) ? std::stoi(argv[3]) : 0;

    TProblemData data;
    ReadData(instance_file, data);

    std::cout << "Instance: " << instance_file << " (Jobs: " << data.num_jobs << ", Machines: " << data.num_machines << ")" << std::endl;
    std::cout << "Chromosome size: " << data.n << std::endl;

    // Fixed seeds for reproducibility in this simple test
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    TSol best_sol;
    best_sol.ofv = std::numeric_limits<double>::max();
    best_sol.rk.resize(data.n);

    auto start_time = std::chrono::high_resolution_clock::now();
    long long iterations = 0;

    data.start_time = omp_get_wtime();
    data.max_time = time_limit;

    while (true) {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time;
        if (elapsed.count() >= time_limit) break;

        TSol s;
        s.rk.resize(data.n);
        for (int i = 0; i < data.n; ++i) s.rk[i] = dist(rng);

        s.ofv = Decoder(s, data);

        if (s.ofv < best_sol.ofv) {
            best_sol = s;
            std::cout << "Iteration " << iterations << " - New best OFV: " << (int)best_sol.ofv 
                      << " (Total Load: " << (best_sol.ofv - (int)best_sol.ofv) * 1e6 << ")" << std::endl;
        }
        iterations++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_time - start_time;

    std::cout << "\n--- Final Results ---" << std::endl;
    PrintSolution(best_sol, data);
    
    double gap = 0;
    if (bks > 0) gap = (best_sol.ofv - bks) * 100.0 / bks;

    // Format for easy shell parsing
    // FINAL_RESULT;instance;ofv;gap;iterations;time
    std::cout << "FINAL_RESULT;" << instance_file << ";" << (int)best_sol.ofv << ";" << gap << ";" << iterations << ";" << total_duration.count() << std::endl;

    FreeMemoryProblem(data);
    return 0;
}
