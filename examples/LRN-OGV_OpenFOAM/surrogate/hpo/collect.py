"""
Collects a campaign's trials into <campaign>/results.csv: one row per trial with its options (trial.json or
config.json) and its outcome (summary.json: best validation QoI error per OP, val MSE, epochs, time; pruned trials
included with their pruning epoch). Also prints the best 15 by validation loss error.

Usage: python hpo/collect.py <campaign_dir>
"""
import glob
import json
import os
import sys

import pandas as pd


def main():
    camp = os.path.abspath(sys.argv[1])
    rows = []
    for d in sorted(glob.glob(os.path.join(camp, "trials", "*"))):
        cfg_f = os.path.join(d, "config.json")
        if not os.path.isfile(cfg_f):
            continue
        cfg = json.load(open(cfg_f))
        row = {"trial": os.path.basename(d)}
        if os.path.isfile(os.path.join(d, "trial.json")):
            row["factor"] = json.load(open(os.path.join(d, "trial.json"))).get("factor", "")
        row.update({k: cfg.get(k) for k in ("arch", "hidden", "layers", "omega", "omega0", "activation", "fourier_features",
                                            "fourier_scale", "no_resnet", "mod_activation", "coords", "mp_aug", "lr", "scheduler",
                                            "batch", "epochs", "weight_decay", "max_grad_norm", "seed")})
        s_f = os.path.join(d, "summary.json")
        if os.path.isfile(s_f):
            s = json.load(open(s_f))
            row.update(status="pruned" if "pruned_at_epoch" in s else "done", pruned_at=s.get("pruned_at_epoch"),
                       val_w_relmae=s.get("best_val_w_relmae"), val_mse=s.get("best_val_mse"),
                       **{k.replace("best_qoi_", ""): s.get(k) for k in s if k.startswith("best_qoi_val")},
                       qoi_epoch=s.get("best_qoi_epoch"), n_params=s.get("n_params"), train_min=(s.get("train_wall_s") or 0) / 60,
                       epoch_s=s.get("mean_epoch_s"))
        else:
            row["status"] = "running or failed"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(camp, "results.csv"), index=False)
    if "val_w_relmae" in df:
        cols = [c for c in ("trial", "factor", "status", "val_w_relmae", "val_w_relmae_ADP", "val_w_relmae_OP1", "val_w_relmae_OP2",
                            "val_mse", "train_min") if c in df]
        print(df[df.status == "done"].sort_values("val_w_relmae")[cols].head(15).round(3).to_string(index=False))
    print(f"{len(df)} trials: {df.status.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
