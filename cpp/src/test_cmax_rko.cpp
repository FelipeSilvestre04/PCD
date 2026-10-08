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

// Include the decoder (it needs TSol and TProblemData)
#include "Problem/Problem_DynamicWindowGCI_Cmax_NoPanic.h"

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

    printf("Starting %s for %.1fs. BKS: %d\n", instance_path, time_limit, bks);

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
            printf("[%.1fs] New Best: %.0f (Gap: %.2f%%) | Iter: %lld\n", elapsed, bestSol.ofv, gap, iterations);
        }
        iterations++;
    }

    double final_gap = (bestSol.ofv - bks) / bks * 100.0;
    printf("\nFINAL_RESULT;%s;%.0f;%.2f;%lld;%.1f\n", instance_path, bestSol.ofv, final_gap, iterations, time_limit);
    
    // Imprime a sequência associada à melhor solução encontrada
    PrintSolution(bestSol, data);

    FreeMemoryProblem(data);
    return 0;
}
