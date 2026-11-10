# HPO campaign: plan and Jean Zay layout

(Merged on 4 Nov from hpo/README.md and hpo/README.md. Results: `../hpo_results/<campaign>/`, log of findings: `../RESULTS.md`.)

## INR hyperparameter campaign (Jean Zay)

**Goal.** Find the INR configuration that best predicts the **QoIs** (mixed-out loss, outflow angle) of unseen blades.
The training loss stays a plain MSE on all cells.
**Fixed:** `data_mean/` (time-averaged fields, medium mesh: the method is developed now and transferred later to the wall×2 DoE), the splits (703 train / 88 val / 88 test), T_geom + T_bal, outputs (p, ρ, Ux, Uy, νt).
**Selection metric:** the validation loss error (mean over the 3 OPs, best over epochs, from MP1/MP2 every 5 epochs). The test set is used once, at the end.
Background findings so far: `RESULTS.md`. Code: `hpo/space.py` defines the space and is the single source of truth for this table.

## Hyperparameters and ranges
Reference = paper config + periodic embedding (adopted after batch `ideas_9oct`), selected by best validation QoI.

Phase 3 was narrowed after the screening; the ranges actually used are in the docstring of `hpo/space.py: suggest` (ω [0.05, 3], ω0 [0.1, 30], width {48…256}, depth 12–28, lr [1e-3, 1e-2], batch {4096, 8192, 16384}, weight decay [3e-4, 1e-2], MLP SiLU/GELU with Fourier features {0, 32} at σ [0.3, 3]; OneCycle, ReLU modulation and periodic coordinates fixed; 200 epochs). The table below is the original plan.

| hyperparameter | reference | phase 2: screening (one factor at a time, 2 seeds each) | phase 3: Optuna range (original plan) |
|---|---|---|---|
| architecture | SIREN | SIREN, or MLP (below) | {SIREN, MLP} |
| ω (hidden), ω0 (first layer), learnable | 0.1, 0.1 | full 6 × 6 grid of {1e-3, 1e-2, 1e-1, 1, 10, 100} | each log-uniform [1e-3, 1e2] (SIREN) |
| MLP activation / Fourier features (number, scale σ) | – | {GELU, SiLU, tanh} × {none, 32 at σ = 1, 32 at σ = 10} | {GELU, SiLU, tanh} × {0, 16, 32, 64} × σ log [0.5, 50] |
| width | 48 | 32, 64, 96, 128, 256 | {32, 48, 64, 96, 128, 192, 256} |
| depth (layers) | 12 | 4, 8, 16, 24 | integer 4–20 |
| ResNet skips | on | off | {on, off} |
| modulation activation | ReLU | identity | {ReLU, identity} |
| learning rate (peak) | 1e-3 | 1e-4, 3e-4, 3e-3, 1e-2 | log [1e-4, 1e-2] |
| schedule | OneCycle (5 % warm-up) | cosine with 5 % warm-up, constant | {OneCycle, cosine} |
| **epochs** | 100 | 200, 400 | fixed at 200 per study, with early stopping by pruning (below) |
| batch (points) | 16384 | 4096, 8192, 32768, 65536 | {4096, 8192, 16384, 32768, 65536} |
| weight decay | 1e-3 | 0, 1e-4, 1e-2 | {0, 1e-4, 1e-3, 1e-2, 1e-1} |
| gradient clipping (FusedLAMB) | max-norm 1 | off | {1, off} |
| MP-line points | off | on | {off, on} |
| coordinates | periodic | plain (as a control) | periodic |

Phase 2 has 74 arms × 2 seeds = **148 trials** of 100–400 epochs.

**Epochs are varied.** In phase 2 they are a factor (100 / 200 / 400). In phase 3 they are fixed at 200 inside a study, because the learning-rate schedule depends on the total. The pruner stops poor trials early, at 25, 75 or 200 epochs.

Not in the campaign:
- fp64: field errors are ~10⁴ × fp32 rounding, and these GPUs run fp64 slowly. To revisit only if a strong configuration plateaus.
- Conservative output variables: low priority.

## Optuna strategy (phase 3)
- **Sampler: TPE** (Tree-structured Parzen Estimator), `multivariate=True, group=True`.
  - The first 30 trials are random, to explore.
  - After that, TPE splits the finished trials into "good" (best ~10–25 %) and "bad", fits a density to each, l(x) and g(x), and proposes the next configuration where l(x)/g(x) is largest.
  - `group=True` handles the conditional branches: ω only exists for SIREN, Fourier features only for the MLP.
  - Multivariate TPE models interactions, e.g. learning rate × batch.
- **Pruner: successive halving** (asynchronous, ASHA-style).
  - Every trial reports its validation QoI error every 5 epochs.
  - At epochs 25, 75 and 200 (rungs, factor 3), a trial continues only if it is in the best third of the trials that reached that rung. Most poor configurations stop at 25 epochs, so ~2–3× more trials fit in the budget.
- **Parallelism:** many workers (4 per GPU, 4 GPUs per node, several nodes) share one study, stored in an Optuna **journal file** on `$WORK` (lock-safe on Lustre, unlike SQLite).
- **Budget:** ~300 trials, plus a second study around the winner if needed.
- **Seed noise:** each Optuna trial has one seed. The top 10 are re-run with 3 seeds in phase 4 before any conclusion.

## Phases and status
| phase | what | where | estimate | status |
|---|---|---|---|---|
| 0 | pilot: reference config with 1, 4 and 8 trials per H100 (throughput, utilisation, reproducibility vs corrsin) | JZ, 3 × 1 H100, dev | ~2 GPU-h | done: 4 per GPU (78 % busy, 1.8× throughput) |
| 1 | ideas batch: periodic embedding, MP-line points | corrsin | done | done (`RESULTS.md`) |
| 2 | screening, 148 trials | JZ H100, dev (4 GPUs per job, packed) | ~30–50 GPU-h | done 13 Oct (`RESULTS.md`) |
| 3 | Optuna, 300 trials, space narrowed by phase 2 (`hpo/space.py: suggest`) | JZ: 3 H100 t3 jobs (4 GPUs × 4 workers) + 1 A100 t3 job (8 GPUs × 3 workers), 72 workers | ~200–400 GPU-h | done 13 Oct: best val 1.73 % (`RESULTS.md`) |
| 4 | top 10 × 3 seeds (200 ep) + top 3 × 3 seeds (400 ep), then the test set, ensembles (sizes 1–8), comparison with the GP | JZ, 2 H100 t3 jobs | ~100–150 GPU-h | **submitted 2 Nov** (`campaigns/confirm`) |
| 5 | retrain the winner(s) on the wall×2 DoE data | JZ | ~10 GPU-h | when that dataset is built |

## Files on Jean Zay (`$WORK/ogv_inr/`)
```
repo/                                        copy of the beta-aero-optim repository (aero_optim/surrogate + these scripts)
repo/aero_optim/simulator/cascade_qoi.py     loss code
data_mean/                                   dataset (blades, MP-line cache, splits, T_bal)
campaigns/<campaign>/                        pilot/, screening/, optuna_1/, confirm/ ...
    trials.json                              (fixed-list campaigns) the trials to run
    study.journal                            (Optuna campaigns) the shared study
    jobs/      Slurm scripts, slurm_<id>.log, pack_<id>.log, worker logs, gpu_<id>.csv, efficiency.csv
    trials/<trial>/   config.json, trial.json, history.csv, summary.json, best*_model.pt, normalizer.npz,
                      tb/ (TensorBoard scalars + HParams), gpu_usage.csv, train.log
    results.csv                              one row per trial (hpo/collect.py)
```
After each phase:
- `campaigns/<campaign>/` is copied to corrsin as `surrogate_ogv/hpo_results/<campaign>/`. To compare all trials offline: `tensorboard --logdir hpo_results/<campaign>/trials` (the HPARAMS tab gives parallel coordinates).
- A ≤ 10-line section is added to `RESULTS.md`.
- Trial folders are thinned: the models of the non-retained trials are deleted, while `results.csv`, TensorBoard and the histories are kept.

## Efficiency (Jean Zay rule: GPU busy > 50 %)
- Every job logs `nvidia-smi` every 10 s, and its mean utilisation per GPU is appended to `jobs/efficiency.csv`.
- The number of trials per GPU comes from the pilot.
- If a configuration is too small to load the GPU (batch 4096), it is packed more densely.

## Jean Zay plan for the INR campaign (draft; nothing submitted)

## What is ready
- `$WORK/ogv_inr/` (`/lustre/fswork/projects/rech/umy/ugg72ez/ogv_inr`):
  - code: the repository copy `repo/` (`aero_optim/surrogate/`, `examples/LRN-OGV_OpenFOAM/surrogate/`);
  - `repo/aero_optim/simulator/cascade_qoi.py` (the loss code, via `AERO_REPO=$WORK/ogv_inr/repo`);
  - `data_mean/`: 1.9 GB, 900 blades + MP-line cache.
- Environment: `module purge; module load arch/h100; module load pytorch-gpu/py3/2.8.0`. This gives torch 2.8 / CUDA 12.8, apex FusedLAMB, tensorboard, scipy, pandas, sklearn. Checked on the login node; nothing to install.
- Budget, project 104610: H100 3808 h left (of 5000), A100 930 h left (of 1000), shared with Rebecca and Mattia.

## How jobs run
- **One Slurm job = 1 node, 4 H100 GPUs, the dev QoS** (`qos_gpu_h100-dev`, < 2 h).
- `#SBATCH` header from `wizard_slurm_launcher.py`: constraint h100, 24 CPUs per GPU, `--hint=nomultithread`, account `<acct>@h100`.
- **Several trials per GPU.** The model is tiny (30k parameters), so a single training run uses only ~57 % of an RTX PRO 5000 even with CUDA graphs, and less of an H100. A small packer script (`jz_pack.py`, to write) runs K trials per GPU concurrently via `CUDA_VISIBLE_DEVICES`, K = 3–4, decided in the pilot.
- **Efficiency is logged.** Each job records `nvidia-smi` every 10 s. The packer reports mean GPU utilisation per GPU at the end, so the > 50 % rule can be checked and reported.
- **Outputs** go to `$WORK/ogv_inr/hpo_results/<phase>/<trial>/` (same files as on corrsin), plus one `results.csv` per phase. They are copied back to corrsin after each phase.
- **Footprint:** ~10 files per trial; WORK inodes are at 41 % of 500k.

## Steps (each one needs your go)
0. **Pilot** (1 job, 1 H100, dev, ~20 min, ~0.5 GPU-h):
   - the paper configuration, alone and then with 2 and 4 trials on the same GPU;
   - measures epoch time and GPU utilisation, and checks that the results match corrsin for the same seed.
1. **Phase 2 screening** (`hpo/README.md`): ~90 trials × 100 epochs, each factor around the paper configuration, 2 seeds where differences are small.
   - 1 job = 4 GPUs × K trials. About 2–3 dev jobs, **~15–25 GPU-h**.
2. **Phase 3 joint search:** Optuna TPE with successive-halving pruning on the validation QoI error. About 200–300 trials, **~40–80 GPU-h**.
   - The trial database is an Optuna journal file on `$WORK`, not SQLite: SQLite locking is unreliable on Lustre.
3. **Confirmation:** the top 5 configurations × 3 seeds, longer training, then a single test-set evaluation. **~10–20 GPU-h**.
4. When the wall×2 DoE is done: rebuild `data_mean` from it and retrain the final configuration(s).

Total expected: **~70–130 H100 GPU-h**, about 2–3 % of the 104610 H100 allocation.
