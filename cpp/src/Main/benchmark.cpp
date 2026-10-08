#include <bits/stdc++.h>
#include <omp.h>
#include "../Data.h"
thread_local std::mt19937 rng;
std::atomic<bool> stop_execution(false);
std::vector<TSol> pool;
#include "../Problem/Problem_BestFit.h"
struct Deadline {};
struct Metrics {
    unsigned long long evaluations = 0;
    double best = INFINITY, best_time = 0;
    std::vector<double> keys;
    std::vector<std::pair<double,double>> history;
};
Metrics metrics[8];
bool recording = false;
double started = 0, budget = 0;
double Decoder(TSol &s, const TProblemData &data) {
    if (recording && omp_get_wtime() - started >= budget) throw Deadline{};
    double cost = DecoderImpl(s, data);
    if (recording) {
        auto &m = metrics[omp_get_thread_num()];
        ++m.evaluations;
        if (cost < m.best) {
            m.best = cost; m.best_time = omp_get_wtime() - started;
            m.keys = s.rk; m.history.emplace_back(m.best_time, cost);
        }
    }
    return cost;
}
#include "../Output.h"
#include "../Method.h"
#include "../QLearning.h"
#include "../MH/BRKGA.h"
int main(int argc, char **argv) {
    if (argc != 6) return 2;
    TProblemData data;
    ReadData(argv[1], data);
    int workers = std::stoi(argv[2]); budget = std::stod(argv[3]);
    unsigned seed = std::stoul(argv[4]);
    if (workers < 1 || workers > 8) return 2;
    if (budget == 0) {
        TSol s; std::ifstream input(argv[5]); double k;
        while (input >> k) s.rk.push_back(k);
        if (s.rk.size() != static_cast<size_t>(data.n)) return 3;
        std::cout << std::setprecision(17) << DecoderImpl(s,data) << '\n'; return 0;
    }
    double init_start = omp_get_wtime();
    rng.seed(seed); pool.resize(20); CreatePoolSolutions(data, 20);
    auto initial = pool.front();
    TRunData run{}; run.MAXTIME = static_cast<int>(budget); run.MAXRUNS = 1;
    run.strategy = 2; run.control = 0; run.restart = 1; run.sizePool = 20;
    omp_set_dynamic(0); omp_set_num_threads(workers);
    double init_s = omp_get_wtime() - init_start;
    started = omp_get_wtime(); recording = true;
    #pragma omp parallel
    {
        rng.seed(seed + 104729U * (omp_get_thread_num()+1));
        try { BRKGA(run, data); } catch (const Deadline &) {}
    }
    double elapsed = omp_get_wtime()-started;
    Metrics best; best.best = initial.ofv; best.keys = initial.rk;
    unsigned long long evaluations = 0;
    std::vector<std::pair<double,double>> history{{0,initial.ofv}};
    for (int i=0; i<workers; ++i) {
        evaluations += metrics[i].evaluations;
        if (metrics[i].best < best.best) best = metrics[i];
        history.insert(history.end(),metrics[i].history.begin(),metrics[i].history.end());
    }
    std::sort(history.begin(),history.end());
    std::ofstream out(argv[5]); out << std::setprecision(17);
    out << "{\"language\":\"cpp\",\"workers\":" << workers
        << ",\"budget_s\":" << budget << ",\"elapsed_s\":" << elapsed
        << ",\"initialization_s\":" << init_s << ",\"evaluations\":" << evaluations
        << ",\"evals_per_s\":" << evaluations/elapsed << ",\"best_cost\":" << best.best
        << ",\"time_to_best_s\":" << best.best_time << ",\"keys\":[";
    for (size_t j=0;j<best.keys.size();++j) { if(j)out<<',';out<<best.keys[j]; }
    out << "],\"history\":["; double current = INFINITY; bool comma=false;
    for (auto [t,c]:history) if(c<current) {
        if(comma)out<<','; comma=true; out<<'['<<t<<','<<c<<']'; current=c;
    }
    out << "]}";
}
