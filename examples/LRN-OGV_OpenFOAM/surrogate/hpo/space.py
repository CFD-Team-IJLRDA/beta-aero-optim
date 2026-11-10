"""
Search space of the INR campaign (single source of truth for HPO_PLAN.md's tables).

DEFAULT: the current reference configuration (paper config + the periodic embedding adopted after batch ideas_9oct).
screening(): phase-2 trial list, one factor at a time around DEFAULT, each with SEEDS seeds.
suggest(trial): phase-3 Optuna space (conditional on the architecture).
Every trial is a dict of train_inr.py options; to_argv() turns it into a command line.
"""
import itertools

DEFAULT = dict(arch="siren", hidden=48, layers=12, omega=0.1, omega0=0.1, coords="periodic", mp_aug=0,
               lr=1e-3, scheduler="onecycle", batch=16384, epochs=100, weight_decay=1e-3, max_grad_norm=1.0,
               mod_activation="relu", no_resnet=False)
SEEDS = (0, 1)
OMEGAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

# phase 2: factor -> list of overrides (each override = one arm, run with every seed in SEEDS)
SCREENING = {
    "reference": [{}],
    "omega x omega0": [dict(omega=w, omega0=w0) for w, w0 in itertools.product(OMEGAS, OMEGAS) if (w, w0) != (0.1, 0.1)],
    "width": [dict(hidden=h) for h in (32, 64, 96, 128, 256)],
    "depth": [dict(layers=n) for n in (4, 8, 16, 24)],
    "resnet": [dict(no_resnet=True)],
    "modulation activation": [dict(mod_activation="identity")],
    "MLP": [dict(arch="mlp", activation=a, fourier_features=f, fourier_scale=s)
            for a in ("gelu", "silu", "tanh") for f, s in ((0, 10.0), (32, 1.0), (32, 10.0))],
    "learning rate": [dict(lr=v) for v in (1e-4, 3e-4, 3e-3, 1e-2)],
    "scheduler": [dict(scheduler=v) for v in ("cosine", "constant")],
    "epochs": [dict(epochs=v) for v in (200, 400)],
    "batch": [dict(batch=v) for v in (4096, 8192, 32768, 65536)],
    "weight decay": [dict(weight_decay=v) for v in (0.0, 1e-4, 1e-2)],
    "gradient clipping": [dict(max_grad_norm=0.0)],
    "MP-line points": [dict(mp_aug=1)],
    "coordinates": [dict(coords="plain")],
}


def screening():
    trials = []
    for factor, arms in SCREENING.items():
        for arm in arms:
            cfg = {**DEFAULT, **arm}
            tag = "ref" if not arm else "_".join(f"{k}{v}" for k, v in arm.items())
            for seed in SEEDS:
                trials.append(dict(name=f"{tag}_s{seed}".replace(".", "p").replace("-", "m"), factor=factor, seed=seed, **cfg))
    return trials


def suggest(trial):
    """
    Phase-3 Optuna space (TPE), narrowed by the screening (13 Oct, hpo_results/screening/screening_arms.csv):
    periodic coordinates, OneCycle, ReLU modulation and weight decay > 0 are fixed (alternatives clearly worse);
    omega in [0.05, 3] (>= 10 diverges), omega0 in [0.1, 30]; MLP only with SiLU/GELU and no or low-scale Fourier features
    (scale 10 diverges); capacity and batch ranges centred on what helped (deeper, wider, smaller batch), capped so that
    the costliest trial fits a t3 job when packed 4 per GPU.
    """
    cfg = dict(coords="periodic", epochs=200, scheduler="onecycle", mod_activation="relu")
    cfg["arch"] = trial.suggest_categorical("arch", ["siren", "mlp"])
    if cfg["arch"] == "siren":
        cfg["omega"] = trial.suggest_float("omega", 0.05, 3.0, log=True)
        cfg["omega0"] = trial.suggest_float("omega0", 0.1, 30.0, log=True)
    else:
        cfg["activation"] = trial.suggest_categorical("activation", ["silu", "gelu"])
        cfg["fourier_features"] = trial.suggest_categorical("fourier_features", [0, 32])
        cfg["fourier_scale"] = trial.suggest_float("fourier_scale", 0.3, 3.0, log=True)
    cfg["hidden"] = trial.suggest_categorical("hidden", [48, 64, 96, 128, 192, 256])
    cfg["layers"] = trial.suggest_int("layers", 12, 28)
    cfg["no_resnet"] = trial.suggest_categorical("no_resnet", [False, True])
    cfg["lr"] = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
    cfg["batch"] = trial.suggest_categorical("batch", [4096, 8192, 16384])
    cfg["weight_decay"] = trial.suggest_float("weight_decay", 3e-4, 1e-2, log=True)
    cfg["max_grad_norm"] = trial.suggest_categorical("max_grad_norm", [1.0, 0.0])
    cfg["mp_aug"] = trial.suggest_categorical("mp_aug", [0, 1])
    cfg["seed"] = trial.number
    return cfg


def suggest_wide(trial):
    """
    Phase-3b space (random search and genetic algorithm, no pruning; see HPO_PLAN.md): the narrowed space of
    suggest() plus the choices phase 3 had fixed, so that their effect can be read from the final results:
    scheduler (OneCycle / cosine / constant), epochs (100 / 200 / 400), lr from 1e-4 to 1e-2 (phase 3 never went below 1e-3
    and pruned high lr), layers from 4 (phase 3: 12-28), modulation activation. Everything else as in suggest().
    """
    cfg = dict(coords="periodic")
    cfg["arch"] = trial.suggest_categorical("arch", ["siren", "mlp"])
    if cfg["arch"] == "siren":
        cfg["omega"] = trial.suggest_float("omega", 0.05, 3.0, log=True)
        cfg["omega0"] = trial.suggest_float("omega0", 0.1, 30.0, log=True)
    else:
        cfg["activation"] = trial.suggest_categorical("activation", ["silu", "gelu"])
        cfg["fourier_features"] = trial.suggest_categorical("fourier_features", [0, 32])
        cfg["fourier_scale"] = trial.suggest_float("fourier_scale", 0.3, 3.0, log=True)
    cfg["hidden"] = trial.suggest_categorical("hidden", [48, 64, 96, 128, 192, 256])
    cfg["layers"] = trial.suggest_int("layers", 4, 28)
    cfg["no_resnet"] = trial.suggest_categorical("no_resnet", [False, True])
    cfg["mod_activation"] = trial.suggest_categorical("mod_activation", ["relu", "identity"])
    cfg["scheduler"] = trial.suggest_categorical("scheduler", ["onecycle", "cosine", "constant"])
    cfg["epochs"] = trial.suggest_categorical("epochs", [100, 200, 400])
    cfg["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    cfg["batch"] = trial.suggest_categorical("batch", [4096, 8192, 16384])
    cfg["weight_decay"] = trial.suggest_float("weight_decay", 3e-4, 1e-2, log=True)
    cfg["max_grad_norm"] = trial.suggest_categorical("max_grad_norm", [1.0, 0.0])
    cfg["mp_aug"] = trial.suggest_categorical("mp_aug", [0, 1])
    cfg["seed"] = trial.number
    return cfg


OPTION = dict(hidden="--hidden", layers="--layers", omega="--omega", omega0="--omega0", arch="--arch", activation="--activation",
              fourier_features="--fourier-features", fourier_scale="--fourier-scale", coords="--coords", mp_aug="--mp-aug",
              lr="--lr", scheduler="--scheduler", batch="--batch", epochs="--epochs", weight_decay="--weight-decay",
              max_grad_norm="--max-grad-norm", mod_activation="--mod-activation", seed="--seed")


def to_argv(cfg):
    argv = []
    for k, v in cfg.items():
        if k == "no_resnet":
            argv += ["--no-resnet"] if v else []
        elif k in OPTION:
            argv += [OPTION[k], str(v)]
    return argv


if __name__ == "__main__":
    t = screening()
    print(len(t), "screening trials;", {f: sum(1 for x in t if x["factor"] == f) for f in SCREENING})
