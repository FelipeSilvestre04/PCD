#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/cpp"
g++ -std=c++20 -O3 -fopenmp src/Main/benchmark.cpp -o benchmark
