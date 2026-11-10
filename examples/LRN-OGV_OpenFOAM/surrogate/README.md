# Neural-field surrogate of the LRN-OGV cascade: scripts

The library (morphing, datasets, networks, predictor, torch geometry) is `aero_optim/surrogate/`; this folder holds the scripts
that build the datasets, train and evaluate the networks, optimise on the surrogate, check the results with CFD and run the
hyperparameter search. Method and results: `beta-aero-optim-runs/report_2026_11/report.pdf` (Sec. 4).

## Code and data are separate
- **Code**: here and in `aero_optim/surrogate/`. Every script starts with `import _repo`, which puts the repository on
  `sys.path`, so nothing needs to be installed in the PyTorch environment.
- **Data** (datasets `data*/`, budget subsets `data_subsets/`, trained models `runs/`, optimisation runs `sopt/`, HPO results,
  reports): the work folder, by default `$AERO_RUNS/surrogate_ogv` with `AERO_RUNS=/mnt/Data_B/beta-aero-optim-runs`
  (`aero_optim/surrogate/paths.py`). Override with `AERO_RUNS` or `SURROGATE_WORK` (e.g. murphy: `AERO_RUNS=~/beta-aero-optim-runs`).
- **Run the scripts from the work folder**: relative arguments such as `--models runs/eff_N200_s0_m0` are relative to it.

## Environments (corrsin)
- `~/miniconda3/envs/ogv_torch`: training, evaluation, surrogate optimisation (PyTorch with CUDA; apex FusedLAMB for training).
- `~/miniconda3/envs/ipynb`: datasets, GP, figures, analysis.
- `~/miniconda3/envs/aero_venv` (+ OpenFOAM v2406 sourced): CFD checks (`verify_cfd.py`).
Extra Python packages over aero_optim: `pip install -e ".[surrogate]"` (torch, scikit-learn, optuna, tensorboard).

## Typical sequence
```bash
S=/mnt/Data_B/beta-aero-optim/examples/LRN-OGV_OpenFOAM/surrogate
cd /mnt/Data_B/beta-aero-optim-runs/surrogate_ogv
PY=~/miniconda3/envs/ogv_torch/bin/python
# 1. dataset from a finished DoE (periodic re-cut, T_geom morphing), then T_bal and the splits
~/miniconda3/envs/ipynb/bin/python $S/build_dataset.py --mean --jobs 32          # -> data_mean/
~/miniconda3/envs/ipynb/bin/python $S/fit_balance.py --data data_mean
# 2. a budget subset of N DoE blades, then 3 networks (ensemble) and their evaluation on the test blades
~/miniconda3/envs/ipynb/bin/python $S/make_data_subset.py --n 200 --seed 0       # -> data_subsets/N200_s0/
$PY $S/train_inr.py --name eff_N200_s0_m0 --data data_subsets/N200_s0 --seed 0 ...   # or a job list: queue_runs.py
$PY $S/eval_inr.py --name eff_N200_s0_m0 --checkpoint best_qoi_model
# 3. NSGA-II on the surrogate (or the GP: --gp), then CFD check of the blades predicted better than the baseline
$PY $S/surrogate_optim.py --models runs/eff_N200_s0_m{0,1,2} --data data_subsets/N200_s0 --out sopt/inr_N200_s0_o1 --seed 1
~/miniconda3/envs/aero_venv/bin/python $S/verify_cfd.py --run sopt/inr_N200_s0_o1 --select better --tag better --budget 48
# 4. gradient-based (SLSQP, epsilon-constraint; --swap: min L_OP s.t. L_ADP <= eps) and its Lagrange multipliers
$PY $S/grad_optim.py --models runs/eff_N200_s0_m{0,1,2} --data data_subsets/N200_s0 --out sopt/grad2_N200_s0_o1
$PY $S/kkt_multipliers.py sopt/grad2_N200_s0_o1
```

## Scripts
| script | what |
|---|---|
| `build_dataset.py`, `fit_balance.py`, `build_mp_aug.py`, `make_data_subset.py` | datasets: per-blade fields in the reference domain, T_bal and splits, measurement-line cache, budget subsets |
| `train_inr.py`, `eval_inr.py`, `queue_runs.py` | training (one GPU or DDP), evaluation (fields and QoIs on test, Pareto, baseline blades), GPU job queue |
| `gp_baseline.py`, `compare_models.py` | scalar GP baseline (Matern 5/2 ARD) and INR-vs-GP QoI metrics |
| `surrogate_optim.py`, `grad_optim.py`, `kkt_multipliers.py` | NSGA-II on the surrogate; SLSQP with autodiff gradients; Lagrange multipliers at the SLSQP optima |
| `verify_cfd.py`, `run_data_efficiency.sh` | CFD check of a surrogate front (cases in `$AERO_RUNS/cfd_sopt/`); data-efficiency pipeline |
| `analyse_data_efficiency.py`, `summarize_batch.py`, `plot_predictions.py`, `plot_transform.py`, `runs_index.py` | analysis, figures, index of trained runs |
| `hpo/` | hyperparameter search on Jean Zay (Optuna workers, fixed trial packs, Slurm job writer, analysis); see `hpo/README.md` |
| `checks/` | one-off diagnostics of the morphing, the datasets and the training step (some use absolute paths of corrsin) |
