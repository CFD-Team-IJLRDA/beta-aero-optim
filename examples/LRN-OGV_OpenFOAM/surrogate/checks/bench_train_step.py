"""
GPU-efficiency benchmark of one INR training step (paper model, 48 x 12, 30k parameters) on one GPU, synthetic data
of the real shape (46 M points). Times the full step (gather batch, forward, backward, FusedLAMB, OneCycle) per variant,
reports ms/step, points/s, epoch time and the GPU busy fraction from torch.profiler (CUDA kernel time / wall time).
--cpu-load N also runs every variant with N busy-spinning background processes: on an idle machine the CPU cores
sit in power-saving states and the launch-bound training loop slows down (observed 21 s vs 10 s per epoch).

Usage: ~/miniconda3/envs/ogv_torch/bin/python checks/bench_train_step.py [--gpu 0] [--steps 200] [--cpu-load 16]
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import torch
from apex.optimizers import FusedLAMB
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts folder: _repo, train_inr, ...
import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
from aero_optim.surrogate.siren import ShiftModulatedSiren  # noqa: E402

N_POINTS = 46_000_000


def spin():
    while True:
        pass


def make(dev):
    torch.manual_seed(0)
    return ShiftModulatedSiren(2, 6, 5, 48, 12, 0.1, 0.1).to(dev)


def run(variant, batch, dev, steps, data):
    model = make(dev)
    opt = FusedLAMB(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=10 * steps, pct_start=0.05)
    fwd = {"eager": model, "compile": torch.compile(model),
           "compile_cudagraphs": torch.compile(model, mode="reduce-overhead")}[variant]
    P, X, Y = data
    perm = torch.randperm(len(Y), device=dev)

    def step(i):
        idx = perm[(i * batch) % (len(Y) - batch):][:batch]
        loss = torch.mean((fwd(P[idx], X[idx]) - Y[idx]) ** 2)
        model.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        return loss

    for i in range(30):                     # warm-up / compilation / graph capture
        step(i)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(steps):
        step(i)
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t) / steps
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t = time.perf_counter()
        for i in range(20):
            step(i)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
    kern = sum(e.device_time_total for e in prof.events() if e.device_type.name == "CUDA")   # all CUDA kernel time [us]
    busy = kern / 1e6 / wall
    return ms, busy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cpu-load", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    data = (torch.randn(N_POINTS, 6, device=dev), torch.rand(N_POINTS, 2, device=dev), torch.randn(N_POINTS, 5, device=dev))
    print(f"{'variant':20s} {'batch':>7s} {'CPU load':>8s} {'ms/step':>8s} {'Mpts/s':>7s} {'epoch [s]':>9s} {'GPU busy':>8s}", flush=True)
    for load in ([0, args.cpu_load] if args.cpu_load else [0]):
        procs = [mp.Process(target=spin, daemon=True) for _ in range(load)]
        for p in procs:
            p.start()
        for variant, batch in [("eager", 16384), ("compile", 16384), ("compile_cudagraphs", 16384),
                               ("compile", 65536), ("compile_cudagraphs", 65536), ("compile", 262144)]:
            ms, busy = run(variant, batch, dev, args.steps, data)
            print(f"{variant:20s} {batch:7d} {load:8d} {ms:8.2f} {batch / ms / 1e3:7.1f} {N_POINTS / batch * ms / 1e3:9.1f} "
                  f"{100 * busy:7.0f}%", flush=True)
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
