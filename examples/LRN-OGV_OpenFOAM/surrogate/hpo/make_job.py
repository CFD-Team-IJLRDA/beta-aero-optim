"""
Writes (and optionally submits) a Slurm script for the INR campaign on Jean Zay, with the header of
wizard_slurm_launcher.py (account umy, constraint h100/a100, QoS dev/t3/t4, --hint=nomultithread) and
CPUs scaled with the GPUs (h100: 24 per GPU, a100: 8 per GPU), because several trials share each GPU.

Modes:
  pack   : python hpo/pack.py <campaign> <trials.json> --per-gpu K   (fixed list: pilot, screening, confirmation)
  optuna : K workers per GPU, python hpo/optuna_worker.py <campaign> --gpu g   (phase 3)
Scripts and Slurm logs go to <campaign>/jobs/.

Usage (on Jean Zay, in $WORK/ogv_inr, with a copy of the repository in repo/):
  python repo/examples/LRN-OGV_OpenFOAM/surrogate/hpo/make_job.py pack campaigns/pilot --trials campaigns/pilot/trials.json --arch h100 --gpus 1 --qos dev --per-gpu 4 [--submit]
"""
import argparse
import os
import subprocess

ARCH = {"h100": dict(constraint="h100", module="arch/h100", cpus_per_gpu=24),
        "a100": dict(constraint="a100", module="arch/a100", cpus_per_gpu=8)}
QOS = {"dev": "01:59:00", "t3": "19:59:00", "t4": "99:59:00"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pack", "optuna"])
    ap.add_argument("campaign")
    ap.add_argument("--trials", help="pack mode: trials JSON")
    ap.add_argument("--arch", default="h100", choices=list(ARCH))
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--qos", default="dev", choices=list(QOS))
    ap.add_argument("--time", default=None, help="override the QoS maximum, e.g. 00:30:00")
    ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--stop-after-min", type=float, default=None, help="default: time limit minus 15 min")
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--sampler", default="tpe", choices=["tpe", "random", "nsga2", "cmaes"], help="optuna mode")
    ap.add_argument("--no-prune", action="store_true", help="optuna mode: run every trial to the end")
    ap.add_argument("--space", default="narrow", choices=["narrow", "wide"], help="optuna mode: search space")
    ap.add_argument("--population", type=int, default=40, help="optuna nsga2 mode: population size")
    ap.add_argument("--name", default=None)
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    camp = os.path.abspath(args.campaign)
    jdir = os.path.join(camp, "jobs")
    os.makedirs(jdir, exist_ok=True)
    a = ARCH[args.arch]
    wall = args.time or QOS[args.qos]
    h, m, _ = (int(x) for x in wall.split(":"))
    stop = args.stop_after_min if args.stop_after_min is not None else max(5, h * 60 + m - 15)
    name = args.name or f"inr_{os.path.basename(camp)}"[:15]
    if args.mode == "pack":
        run = (f"python $SRC/hpo/pack.py {camp} {os.path.abspath(args.trials)} --per-gpu {args.per_gpu} "
               f"--stop-after-min {stop}")
    else:
        workers = "\n".join(f"python $SRC/hpo/optuna_worker.py {camp} --gpu {g} --stop-after-min {stop} --n-trials {args.n_trials} "
                            f"--sampler {args.sampler} {'--no-prune ' if args.no_prune else ''}--space {args.space} --population {args.population} "
                            f"> {jdir}/worker_${{SLURM_JOB_ID}}_g{g}_{k}.log 2>&1 & PIDS+=($!)"
                            for g in range(args.gpus) for k in range(args.per_gpu))
        run = (f"nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits "
               f"-l 10 > {jdir}/gpu_${{SLURM_JOB_ID}}.csv &\nSMI=$!\nPIDS=()\n{workers}\nwait \"${{PIDS[@]}}\"\nkill $SMI")
    script = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output={jdir}/slurm_%j.log
#SBATCH --constraint={a['constraint']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --cpus-per-task={a['cpus_per_gpu'] * args.gpus}
#SBATCH --time={wall}
#SBATCH --qos=qos_gpu_{args.arch}-{args.qos}
#SBATCH --hint=nomultithread
#SBATCH --account=umy@{args.arch}

module purge
module load {a['module']}
module load pytorch-gpu/py3/2.8.0
export SURROGATE_WORK=$WORK/ogv_inr                                   # data, campaigns (aero_optim/surrogate/paths.py)
export SRC=$WORK/ogv_inr/repo/examples/LRN-OGV_OpenFOAM/surrogate    # scripts: a copy of the repository in repo/
export OMP_NUM_THREADS=2
cd $WORK/ogv_inr
echo "job $SLURM_JOB_ID on $SLURMD_NODENAME, {args.gpus} x {args.arch}, mode {args.mode}"
{run}
"""
    path = os.path.join(jdir, f"{name}_{args.mode}.sh")
    open(path, "w").write(script)
    print(f"written {path}")
    if args.submit:
        print(subprocess.run(["sbatch", path], capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    main()
