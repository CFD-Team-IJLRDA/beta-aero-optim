"""
Shift-modulated SIREN (paper App. "Shift-Modulated SIREN"), ported from NO_pipeline/src/models/shift_mod_siren.py.

    h0 = sin(omega_0 (W_in x + b_in))
    h_i = sin(omega (W_i h_{i-1} + b_i + sigma(W_mod,i p + b_mod,i))) + h_{i-1}      (ResNet skip)
    u  = W_out h_L + b_out

Changes w.r.t. the original:
- several output channels in one model (out_features = number of fields);
- input split as forward(params, coords) instead of slicing one concatenated tensor;
- the unused hard velocity constraint and dropout are removed (dropout_rate was 0 in all paper runs).
Initialisation is unchanged: first layer U(-1/fan_in, 1/fan_in), hidden and output layers
U(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega).
"""
import math

import torch
import torch.nn as nn

ACT = {"relu": torch.relu, "identity": lambda t: t, "tanh": torch.tanh, "silu": nn.functional.silu}


class ShiftModulatedSiren(nn.Module):
    def __init__(self, in_features=2, shift_size=6, out_features=5, hidden_features=48, num_hidden_layers=12,
                 omega=0.1, omega_0=0.1, use_resnet=True, mod_activation="relu"):
        super().__init__()
        self.input_layer = nn.Linear(in_features, hidden_features)
        self.hidden = nn.ModuleList([nn.Linear(hidden_features, hidden_features) for _ in range(num_hidden_layers - 1)])
        self.modulation = nn.ModuleList([nn.Linear(shift_size, hidden_features) for _ in range(num_hidden_layers - 1)])
        self.output_layer = nn.Linear(hidden_features, out_features)
        self.omega = nn.Parameter(torch.tensor(float(omega)))
        self.omega_0 = nn.Parameter(torch.tensor(float(omega_0)))
        self.use_resnet = use_resnet
        self.mod_activation = ACT[mod_activation]
        with torch.no_grad():
            self.input_layer.weight.uniform_(-1 / in_features, 1 / in_features)
            for layer in list(self.hidden) + [self.output_layer]:
                bound = math.sqrt(6 / layer.in_features) / float(omega)
                layer.weight.uniform_(-bound, bound)

    def forward(self, params, coords):
        h = torch.sin(self.omega_0 * self.input_layer(coords))
        for lin, mod in zip(self.hidden, self.modulation):
            h_next = torch.sin(self.omega * (lin(h) + self.mod_activation(mod(params))))
            h = h_next + h if self.use_resnet else h_next
        return self.output_layer(h)


MLP_ACT = {"gelu": nn.functional.gelu, "silu": nn.functional.silu, "tanh": torch.tanh, "relu": torch.relu}


class ShiftModulatedMLP(nn.Module):
    """
    Standard-activation alternative to the SIREN, with the same shift modulation and ResNet skips:

        z  = [sin(2 pi x B), cos(2 pi x B)]  (Gaussian Fourier features, B ~ N(0, s^2), fixed)  or  z = x
        h0 = act(W_in z + b_in)
        h_i = act(W_i h_{i-1} + b_i + sigma(W_mod,i p + b_mod,i)) + h_{i-1}
        u  = W_out h_L + b_out

    Without Fourier features a plain MLP is biased towards smooth functions (spectral bias); the features (scale s)
    let it represent the thin wakes and boundary layers. Default PyTorch initialisation.
    """
    def __init__(self, in_features=2, shift_size=6, out_features=5, hidden_features=48, num_hidden_layers=12,
                 activation="gelu", fourier_features=0, fourier_scale=10.0, use_resnet=True, mod_activation="relu", seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        if fourier_features:
            self.register_buffer("B", torch.randn(in_features, fourier_features, generator=g) * fourier_scale)
            n_in = 2 * fourier_features
        else:
            self.B = None
            n_in = in_features
        self.input_layer = nn.Linear(n_in, hidden_features)
        self.hidden = nn.ModuleList([nn.Linear(hidden_features, hidden_features) for _ in range(num_hidden_layers - 1)])
        self.modulation = nn.ModuleList([nn.Linear(shift_size, hidden_features) for _ in range(num_hidden_layers - 1)])
        self.output_layer = nn.Linear(hidden_features, out_features)
        self.act = MLP_ACT[activation]
        self.use_resnet = use_resnet
        self.mod_activation = ACT[mod_activation]

    def forward(self, params, coords):
        z = coords if self.B is None else torch.cat([torch.sin(2 * math.pi * coords @ self.B),
                                                     torch.cos(2 * math.pi * coords @ self.B)], dim=-1)
        h = self.act(self.input_layer(z))
        for lin, mod in zip(self.hidden, self.modulation):
            h_next = self.act(lin(h) + self.mod_activation(mod(params)))
            h = h_next + h if self.use_resnet else h_next
        return self.output_layer(h)


def build_model(cfg, in_features, shift_size=6, out_features=5):
    """Model from a run configuration (config.json / argparse namespace as dict); older configs default to the SIREN."""
    common = dict(in_features=in_features, shift_size=shift_size, out_features=out_features,
                  hidden_features=cfg["hidden"], num_hidden_layers=cfg["layers"], use_resnet=not cfg.get("no_resnet", False),
                  mod_activation=cfg.get("mod_activation", "relu"))
    if cfg.get("arch", "siren") == "mlp":
        return ShiftModulatedMLP(activation=cfg.get("activation", "gelu"), fourier_features=cfg.get("fourier_features", 0),
                                 fourier_scale=cfg.get("fourier_scale", 10.0), seed=cfg.get("seed", 0), **common)
    omega = cfg["omega"]
    omega_0 = cfg.get("omega0") or omega
    return ShiftModulatedSiren(omega=omega, omega_0=omega_0, **common)
