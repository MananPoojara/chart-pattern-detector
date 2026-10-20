#!/bin/bash
# usage: scripts/run_bench.sh "<method>" ...   sequential, low priority, 2 threads (safe on a shared machine)
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CPD_THREADS=2
for m in "$@"; do nice -n 15 python -m cpd.bench run "$m" 2>&1 | grep --line-buffered -v -i warn; done
