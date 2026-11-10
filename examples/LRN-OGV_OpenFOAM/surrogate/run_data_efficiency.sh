#!/bin/bash
# Data-efficiency study: NSGA-II on the surrogate (sopt/<arm>_N<N>_s<k>_o<o>/) then CFD verification (cases in ../cfd_sopt/).
# Fronts: inr|gp x N = 100, 200, 400 x budget subsets s0-s2 x optimiser seeds o1-o2. An INR front waits for its 3 trained models.
# Usage: nohup bash <this folder>/run_data_efficiency.sh BUDGET FRONT... > sopt/pipeline.log 2>&1 &     (FRONT = gp_N100_s0_o2, ...)
#   BUDGET = solver processes at once; 75 = one front (25 blades x 3 OPs) in one wave. Keep corrsin's load < ~100 (128 threads).
#   BUDGET = opt-only: optimise the fronts but do not verify them (verification elsewhere, e.g. murphy).
SRC=$(cd "$(dirname "$0")" && pwd)                                      # scripts (this folder)
cd "${SURROGATE_WORK:-${AERO_RUNS:-/mnt/Data_B/beta-aero-optim-runs}/surrogate_ogv}"   # work folder: data, runs, sopt
export PATH=$HOME/miniconda3/envs/aero_venv/bin:$PATH OMP_NUM_THREADS=4
PYT=$HOME/miniconda3/envs/ogv_torch/bin/python
PYF=$HOME/miniconda3/envs/aero_venv/bin/python
B=$1; shift
for f in "$@"; do
    IFS=_ read arm N s o <<< "$f"; N=${N#N}; s=${s#s}; o=${o#o}
    run=sopt/$f
    if [ ! -f $run/run.json ]; then
        if [ $arm = gp ]; then
            $PYT $SRC/surrogate_optim.py --gp --data data_subsets/N${N}_s$s --out $run --seed $o > /dev/null 2> $run.err || continue
        else
            until [ -f runs/eff_N${N}_s${s}_m0/eval.log ] && [ -f runs/eff_N${N}_s${s}_m1/eval.log ] && [ -f runs/eff_N${N}_s${s}_m2/eval.log ] \
                  && ! pgrep -f "name eff_N${N}_s${s}_m" > /dev/null; do sleep 120; done
            $PYT $SRC/surrogate_optim.py --models runs/eff_N${N}_s${s}_m{0,1,2} --data data_subsets/N${N}_s$s --out $run --seed $o --device cuda:$(( o % 2 )) > /dev/null 2> $run.err || continue
        fi
        rm -f $run.err
    fi
    [ "$B" = opt-only ] && continue
    while pgrep -f "verify_cfd.py --run $run " > /dev/null; do sleep 60; done           # already being verified
    [ -f $run/verification_cfd.csv ] || $PYF $SRC/verify_cfd.py --run $run --k 25 --budget $B > $run/verify.log 2>&1
    echo "$(date '+%F %T') $run done"
done
echo "$(date '+%F %T') all done"
