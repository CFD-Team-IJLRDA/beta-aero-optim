"""
Step 2: train the shift-modulated SIREN on the transformed OGV dataset (one model, all fields, all operating points).

Inputs per cell: params = [c1..c5, inlet angle], coords = [x_hat, y_hat]; targets [p, rho, Ux, Uy, nut] (z-scored).
Defaults = the paper's configuration: 48 features x 12 layers, ResNet skips, omega = omega_0 = 0.1 (learnable),
ReLU on the modulation, LAMB (official apex FusedLAMB, its defaults), lr 1e-3, one-cycle schedule,
batch 16384 points, 100 epochs, MSE loss.
The whole training set lives on each GPU and is reshuffled every epoch; the validation blades are evaluated
every epoch on rank 0 (MSE in normalized space + NRMSE per field in physical units); the best validation model is kept.
The test blades are also evaluated every epoch for monitoring only (never used for model selection).

Multi-GPU (DDP, torchrun): all ranks draw the same permutation and each takes a disjoint share of every global batch,
so --batch stays the global batch and the optimisation matches the single-GPU run (gradients averaged by DDP).

Model and search options (hpo/README.md): --arch siren|mlp, --omega / --omega0, --activation and --fourier-* (mlp),
--no-resnet, --mod-activation, --scheduler onecycle|cosine|constant, --qoi-every, --test-every (0: no test monitoring).
In-process use (Optuna): main(argv, trial) reports the validation QoI error every --qoi-every epochs to the trial
and stops early if the pruner says so; it returns the summary dict.

Usage (env ~/miniconda3/envs/ogv_torch):
  python train_inr.py --name run01 [--epochs 100] [--train-subset N] [--gpu 0]      # one GPU
  torchrun --standalone --nproc_per_node 2 train_inr.py --name run01                # both GPUs
Outputs in runs/<name>/: config.json, normalizer.npz, best_model.pt, history.csv (per epoch), loss.png,
best_qoi_model.pt (best validation QoI error), summary.json (best epochs, timings), tb/ (TensorBoard: tensorboard --logdir runs),
gpu_usage.csv (nvidia-smi every 2 s during the run: utilization, memory, power; GPU efficiency check)
"""
import argparse
import json
import os
import sys
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from apex.optimizers import FusedLAMB  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
from aero_optim.surrogate.data import (COORDS, INLET_ANGLE, TARGETS, Normalizer, balance, embed, load_balance, load_blade,  # noqa: E402
                                       raw_arrays, set_data, splits, to_device)
from aero_optim.simulator.cascade_qoi import plane_qois  # noqa: E402
from aero_optim.surrogate.siren import build_model  # noqa: E402

HERE = _repo.WORK                 # data, runs, sopt: the surrogate work folder (aero_optim/surrogate/paths.py)


OPS = ["ADP", "OP1", "OP2"]


class ValQoI:
    """
    Validation QoI error from the measurement lines only: the network is evaluated on MP1/MP2 (1001 points each) of
    every validation blade and OP, the mixed-out loss and outflow angle are computed as in the CFD post-processing,
    and compared with the CFD time-averaged QoIs. Cheap (a few 1e5 points), done every 5 epochs.
    """
    def __init__(self, names, bal, coords, norm, dev):
        P, X, self.q_true = [], [], []
        for name in names:
            d = load_blade(name)
            x = embed(balance(np.vstack([d["mp1_ref"], d["mp2_ref"]]), bal), coords, bal)
            for k in range(3):
                P.append(np.tile(np.r_[d["pod"], INLET_ANGLE[k]], (len(x), 1)))
                X.append(x)
            self.q_true.append(d["qoi"])
        self.n_line = len(d["mp1_ref"])
        t = lambda a, key: torch.from_numpy(norm(key, np.vstack(a)).astype(np.float32)).to(dev)
        self.p, self.x, self.norm = t(P, "P"), t(X, "X"), norm
        self.q_true = np.array(self.q_true)                        # (blades, 3 OPs, 2)

    @torch.no_grad()
    def __call__(self, fwd):
        y = self.norm.inverse_Y(torch.cat([fwd(self.p[i:i + 262144], self.x[i:i + 262144])
                                           for i in range(0, len(self.p), 262144)]).cpu().numpy())
        y = y.reshape(len(self.q_true), 3, 2, self.n_line, -1)
        df = lambda f: pd.DataFrame({"p": f[:, 0], "rho": f[:, 1], "Ux": f[:, 2], "Uy": f[:, 3]})
        q = np.array([[[plane_qois(df(y[b, k, 0]), df(y[b, k, 1]))[c] for c in ("MixedoutLossCoef", "OutflowAngle")]
                       for k in range(3)] for b in range(len(self.q_true))])
        rel = np.abs(q[..., 0] / self.q_true[..., 0] - 1).mean(0) * 100
        out = {f"val_w_relmae_{op}": float(v) for op, v in zip(OPS, rel)}
        out["val_w_relmae"] = float(rel.mean())
        out["val_phi_mae"] = float(np.abs(q[..., 1] - self.q_true[..., 1]).mean())
        return out


def nrmse_per_channel(y, y_hat, blade):
    """Paper metric: per blade (and OP) RMSE / range, averaged over blades; physical units; per channel."""
    out = []
    for b in np.unique(blade):
        m = blade == b
        rng = y[m].max(0) - y[m].min(0)
        out.append(np.sqrt(((y[m] - y_hat[m]) ** 2).mean(0)) / (rng + 1e-12))
    return np.mean(out, axis=0)


HPARAMS = ("arch", "hidden", "layers", "omega", "omega0", "activation", "fourier_features", "fourier_scale", "no_resnet",
           "mod_activation", "coords", "mp_aug", "lr", "scheduler", "batch", "epochs", "weight_decay", "max_grad_norm",
           "optimizer", "seed", "train_subset", "data")


def log_hparams(tb, args, metrics):
    """TensorBoard HParams entry (compare all runs of a campaign in the HPARAMS tab)."""
    hp = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in vars(args).items() if k in HPARAMS}
    tb.add_hparams(hp, {f"hparam/{k}": float(v) for k, v in metrics.items()}, run_name="hparams")


_CACHE = {}


def arrays(names, bal, coords, mp_aug=0):
    """raw_arrays, memoised in-process (Optuna workers run many trials in one process: load the blades once)."""
    from aero_optim.surrogate import data as D
    key = (D.DATA, tuple(names), coords, mp_aug)
    if key not in _CACHE:
        _CACHE[key] = raw_arrays(names, bal, coords, mp_aug)
    return _CACHE[key]


class Pruned(Exception):
    """Raised (after cleaning up) when an Optuna trial is pruned; the caller converts it to optuna.TrialPruned."""


def main(argv=None, trial=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--omega", type=float, default=0.1, help="SIREN hidden-layer frequency (learnable)")
    ap.add_argument("--omega0", type=float, default=None, help="SIREN first-layer frequency (learnable); default = --omega")
    ap.add_argument("--arch", default="siren", choices=["siren", "mlp"])
    ap.add_argument("--activation", default="gelu", choices=["gelu", "silu", "tanh", "relu"], help="mlp only")
    ap.add_argument("--fourier-features", type=int, default=0, help="mlp only: number of Gaussian Fourier features (0 = none)")
    ap.add_argument("--fourier-scale", type=float, default=10.0, help="mlp only: std of the Fourier frequencies")
    ap.add_argument("--no-resnet", action="store_true", help="no residual skips")
    ap.add_argument("--mod-activation", default="relu", choices=["relu", "identity", "tanh", "silu"])
    ap.add_argument("--scheduler", default="onecycle", choices=["onecycle", "cosine", "constant"])
    ap.add_argument("--qoi-every", type=int, default=5, help="validation QoI (and field NRMSE) every N epochs")
    ap.add_argument("--test-every", type=int, default=1, help="test-set MSE every N epochs (monitoring only); 0 = never")
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"), help="parent folder of runs/<name>")
    ap.add_argument("--optimizer", default="lamb", choices=["lamb", "adamw"])
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="FusedLAMB global gradient-norm clipping (apex default 1.0); <= 0 disables it "
                         "(apex divides by norm/max, so 0 itself would zero every gradient)")
    ap.add_argument("--train-subset", type=int, default=None, help="use only the first N training blades (data-efficiency study)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data", help="data (last-iteration fields) or data_mean (time-averaged fields)")
    ap.add_argument("--coords", default="plain", choices=list(COORDS),
                    help="coordinate embedding: plain (x_hat, y_hat) or periodic (blade-aware periodic in y_hat, aero_optim/surrogate/data.embed)")
    ap.add_argument("--mp-aug", type=int, default=0,
                    help="add the CFD values interpolated on the MP1/MP2 lines of each training blade, repeated N times "
                         "(build_mp_aug.py); 0 = off")
    ap.add_argument("--compile-mode", default="reduce-overhead",
                    help="torch.compile mode; reduce-overhead = CUDA graphs: the step is launch-bound (30k parameters), "
                         "2-2.7x faster at batch 16384 (checks/bench_train_step.py); 'default' = plain compile")
    args = ap.parse_args(argv)
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
        dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(dev)
    else:
        dev = torch.device(f"cuda:{args.gpu}")
    main_rank = rank == 0
    log = print if main_rank else (lambda *a, **k: None)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = os.path.join(args.out_root, args.name)
    os.makedirs(out, exist_ok=True)

    set_data(args.data)
    sp, bal = splits(), load_balance()
    train_names = sp["train"][: args.train_subset] if args.train_subset else sp["train"]
    t0 = time.time()
    Ptr, Xtr, Ytr, _, _ = arrays(train_names, bal, args.coords, args.mp_aug)
    Pva, Xva, Yva, Bva, Ova = arrays(sp["val"], bal, args.coords)
    norm = Normalizer(Ptr, Xtr, Ytr)
    if main_rank:
        norm.save(os.path.join(out, "normalizer.npz"))
    ptr, xtr, ytr = to_device(norm, Ptr, Xtr, Ytr, dev)
    pva, xva, yva = to_device(norm, Pva, Xva, Yva, dev)
    if main_rank and args.test_every > 0:
        Pte, Xte, Yte, _, _ = arrays(sp["test"], bal, args.coords)
        pte, xte, yte = to_device(norm, Pte, Xte, Yte, dev)
    t_load = time.time() - t0
    log(f"data: {len(train_names)} train blades, {len(ytr) / 1e6:.1f} M points; {len(sp['val'])} val blades, "
          f"{len(yva) / 1e6:.1f} M points; loaded in {time.time() - t0:.0f} s", flush=True)
    if main_rank:
        json.dump(vars(args) | {"train_blades": len(train_names), "targets": TARGETS, "gpus": world,
                                "optimizer_impl": "apex.optimizers.FusedLAMB" if args.optimizer == "lamb" else "torch AdamW"},
                  open(os.path.join(out, "config.json"), "w"), indent=1)

    model = build_model(vars(args), COORDS[args.coords], ptr.shape[1], ytr.shape[1]).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    if args.optimizer == "lamb":
        opt = FusedLAMB(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(ytr) // args.batch + (len(ytr) % args.batch >= world)    # no empty shard on the last batch
    total = steps_per_epoch * args.epochs
    if args.scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    elif args.scheduler == "cosine":         # 5 % linear warm-up, then cosine decay to 0
        warm = max(1, int(0.05 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    fwd = torch.compile(DDP(model, device_ids=[dev.index]) if world > 1 else model, mode=args.compile_mode)
    fwd_eval = torch.compile(model)
    gen = torch.Generator(device=dev)
    log(f"model: {n_params} parameters; {steps_per_epoch} steps / epoch; {world} GPU(s)", flush=True)

    tb = SummaryWriter(os.path.join(out, "tb")) if main_rank else None
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw",
                            "--format=csv", "-l", "2"], stdout=open(os.path.join(out, "gpu_usage.csv"), "w"),
                           stderr=subprocess.DEVNULL) if main_rank else None
    hist, best, best_ep = [], np.inf, 0
    best_q, best_q_ep = np.inf, 0
    val_qoi = ValQoI(sp["val"], bal, args.coords, norm, dev) if main_rank else None
    t_train = time.time()
    for ep in range(1, args.epochs + 1):
        te = time.time()
        model.train()
        gen.manual_seed(args.seed * 100003 + ep)                    # same permutation on every rank
        perm = torch.randperm(len(ytr), device=dev, generator=gen)
        tot = torch.zeros((), device=dev)
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch:(i + 1) * args.batch][rank::world]
            loss = torch.mean((fwd(ptr[idx], xtr[idx]) - ytr[idx]) ** 2)
            model.zero_grad(set_to_none=True)          # apex FusedLAMB.zero_grad has no set_to_none
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.detach() * len(idx)
        if world > 1:
            dist.all_reduce(tot)
        train_loss = float(tot) / len(ytr)
        if not main_rank:
            dist.barrier()
            continue
        model.eval()
        with torch.no_grad():
            pred = torch.cat([fwd_eval(pva[i:i + 262144], xva[i:i + 262144]) for i in range(0, len(yva), 262144)])
            val_loss = float(torch.mean((pred - yva) ** 2))
            test_loss = float(sum(torch.sum((fwd_eval(pte[i:i + 262144], xte[i:i + 262144]) - yte[i:i + 262144]) ** 2)
                                  for i in range(0, len(yte), 262144)) / yte.numel()) \
                if args.test_every > 0 and ep % args.test_every == 0 else np.nan
        row = {"epoch": ep, "train_mse": train_loss, "val_mse": val_loss, "test_mse": test_loss, "lr": sched.get_last_lr()[0],
               "omega": float(getattr(model, "omega", np.nan)), "omega_0": float(getattr(model, "omega_0", np.nan)),
               "time_s": time.time() - te}
        if ep % args.qoi_every == 0 or ep == args.epochs:
            nr = nrmse_per_channel(Yva, norm.inverse_Y(pred.cpu().numpy()), Bva * 3 + Ova)
            row.update({f"val_nrmse_{t}": v for t, v in zip(TARGETS, nr)})
            row.update(val_qoi(fwd_eval))
            if row["val_w_relmae"] < best_q:
                best_q, best_q_ep = row["val_w_relmae"], ep
                torch.save(model.state_dict(), os.path.join(out, "best_qoi_model.pt"))
            if trial is not None:
                trial.report(row["val_w_relmae"], ep)
                if trial.should_prune():
                    hist.append(row)
                    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
                    smi.terminate()
                    log_hparams(tb, args, {"val_w_relmae": best_q, "val_mse": best, "pruned_at_epoch": ep})
                    tb.close()
                    json.dump({"pruned_at_epoch": ep, "best_val_mse": best, "best_val_w_relmae": best_q,
                               "best_qoi_epoch": int(best_q_ep), "train_wall_s": time.time() - t_train},
                              open(os.path.join(out, "summary.json"), "w"), indent=1)
                    raise Pruned(f"pruned at epoch {ep}")
        hist.append(row)
        for k, v in row.items():
            if k != "epoch":
                tb.add_scalar(k.replace("_mse", "/mse").replace("val_nrmse_", "val_nrmse/").replace("val_w_", "val_qoi/w_")
                              .replace("val_phi_", "val_qoi/phi_"), v, ep)
        if val_loss < best:
            best, best_ep = val_loss, ep
            torch.save(model.state_dict(), os.path.join(out, "best_model.pt"))
        print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
        if world > 1:
            dist.barrier()

    if world > 1:
        dist.destroy_process_group()
    if not main_rank:
        return
    smi.terminate()
    h = pd.DataFrame(hist)
    b = h.set_index("epoch").loc[best_ep]
    q = h.set_index("epoch").loc[best_q_ep]
    summary = {"best_epoch": int(best_ep), "best_val_mse": best,
               "best_qoi_epoch": int(best_q_ep), "best_val_w_relmae": best_q,
               **{f"best_qoi_{k}": float(q[k]) for k in ("val_w_relmae_ADP", "val_w_relmae_OP1", "val_w_relmae_OP2", "val_phi_mae")},
               "train_mse_at_best": b.train_mse, "test_mse_at_best": b.test_mse,
               "final_epoch_val_mse": float(h.val_mse.iloc[-1]), "n_params": n_params, "gpus": world,
               "data_load_s": t_load, "train_wall_s": time.time() - t_train, "mean_epoch_s": float(h.time_s.mean()),
               "gpu": torch.cuda.get_device_name(dev)}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    log_hparams(tb, args, {"val_w_relmae": best_q, "val_mse": best, **{k: summary[f"best_qoi_{k}"] for k in
                ("val_w_relmae_ADP", "val_w_relmae_OP1", "val_w_relmae_OP2", "val_phi_mae")}, "train_min": summary["train_wall_s"] / 60})
    tb.close()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(h.epoch, h.train_mse, label=f"train ({len(train_names)} blades)", color="#2a78d6")
    ax.semilogy(h.epoch, h.val_mse, label=f"validation ({len(sp['val'])} blades)", color="#eb6834")
    if h.test_mse.notna().any():
        ax.semilogy(h.epoch, h.test_mse, label=f"test ({len(sp['test'])} blades, monitoring only)", color="#1baf7a", lw=0.9)
    ax.axvline(best_ep, color="#52514e", lw=0.8, ls=":")
    ax.text(best_ep, ax.get_ylim()[1], f" best val: epoch {best_ep}", va="top", fontsize=8, color="#52514e")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (normalized fields)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title(f"{args.name}: {n_params} parameters, {len(train_names)} training blades", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "loss.png"), dpi=600)
    plt.close(fig)
    print(f"done in {(time.time() - t0) / 60:.1f} min, best val MSE {best:.4g}")
    return summary


if __name__ == "__main__":
    main()
