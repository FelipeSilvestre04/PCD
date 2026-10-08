#include "Problem.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

struct TSol {
    std::vector<double> rk;
    double ofv;
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./runner [instance_path] [time_limit]" << std::endl;
        return 1;
    }

    char* instance_path = argv[1];
    double time_limit = std::atof(argv[2]);

    TProblemData data;
    ReadData(instance_path, data);

    TSol sol;
    sol.rk.resize(data.n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto start_time = std::chrono::high_resolution_clock::now();
    double best_cost = std::numeric_limits<double>::infinity();
    unsigned long long evaluations = 0;

    std::cout << ">> Running strictly Problem.h logic..." << std::endl;

    while (true) {
        for(int i=0; i<data.n; ++i) sol.rk[i] = dist(rng);
        
        double cost = Decoder(sol, data);
        
        if (cost < best_cost) best_cost = cost;
        evaluations++;

        if (evaluations % 1000 == 0) {
             auto now = std::chrono::high_resolution_clock::now();
             std::chrono::duration<double> elapsed = now - start_time;
             if (elapsed.count() >= time_limit) break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = end_time - start_time;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Instance: " << instance_path << std::endl;
    std::cout << "Time: " << total_time.count() << "s" << std::endl;
    std::cout << "Best Cost: " << best_cost << std::endl;
    std::cout << "Evaluations: " << evaluations << std::endl;
    std::cout << "Speed: " << (evaluations / total_time.count()) << " sol/s" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    return 0;
}
