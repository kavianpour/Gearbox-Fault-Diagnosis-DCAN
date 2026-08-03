"""
Training / evaluation driver for the DCAN gearbox fault diagnosis model.

Runs every requested method on both transfer tasks (A -> B and B -> A),
repeats each experiment `cfg.runs` times (Section III.B) and reports the mean
target-domain classification accuracy.

The objective optimised for the proposed method is Eq. 6:

    L = L_c + alpha * L_d + beta * L_coral

with L_c the source classification loss, L_d the adversarial domain
discriminator loss, and L_coral the correlation alignment loss on FC2.

Examples:
    python train.py                              # full comparison table
    python train.py --methods DCAN --runs 3
    python train.py --tasks A2B --epochs 50 --device cpu
"""

import os
import csv
import copy
import json
import argparse
import itertools

import numpy as np
import torch
import torch.nn as nn

import data as D
import classic
import da_losses
from config import cfg as default_cfg
from model import DCAN, calc_coeff, xavier_init
from methods import DEEP_METHODS, PROPOSED


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _evaluate(net, loader, device):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            _, logits = net(x.to(device))
            correct += (logits.argmax(dim=1).cpu() == y).sum().item()
            total += y.numel()
    net.train()
    return 100.0 * correct / max(total, 1)


# --------------------------------------------------------------------------- #
# One deep training run
# --------------------------------------------------------------------------- #
def run_deep(spec, cfg, source, target, seed):
    _set_seed(seed)
    loaders = D.transfer_loaders(cfg, source, target, seed)
    device = torch.device(cfg.device)

    net = DCAN(cfg, spec).to(device)
    xavier_init(net)

    optimiser = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=cfg.lr_gamma)
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    da_loss_fn = da_losses.get_da_loss(spec.da_loss)

    step = 0
    net.train()
    for epoch in range(cfg.epochs):
        adapting = epoch >= cfg.warmup_epochs
        target_iter = itertools.cycle(loaders["target_train"])

        for xs, ys in loaders["source_train"]:
            xt, _ = next(target_iter)
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

            feat_s, logits_s = net(xs)
            loss = class_criterion(logits_s, ys)                       # Eq. 1

            if adapting:
                feat_t, _ = net(xt)

                if da_loss_fn is not None:                             # Eq. 2
                    loss = loss + cfg.beta * da_loss_fn(feat_s, feat_t)

                if spec.use_adversarial:                               # Eq. 5
                    coeff = calc_coeff(step, alpha=cfg.grl_alpha,
                                       max_iter=cfg.grl_max_iter)
                    d_s = net.discriminator(feat_s, coeff)
                    d_t = net.discriminator(feat_t, coeff)
                    domain_pred = torch.cat([d_s, d_t], dim=0)
                    domain_true = torch.cat([
                        torch.ones_like(d_s), torch.zeros_like(d_t)], dim=0)
                    loss = loss + cfg.alpha * domain_criterion(domain_pred, domain_true)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            step += 1

        scheduler.step()

    return _evaluate(net, loaders["target_test"], device)


def run_classic(name, cfg, source, target, seed):
    _set_seed(seed)
    loaders = D.transfer_loaders(cfg, source, target, seed)
    return classic.CLASSIC_METHODS[name](loaders["source_train"], loaders["target_test"])


# --------------------------------------------------------------------------- #
# Experiment loop
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None, help="e.g. A2B B2A")
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = copy.copy(default_cfg)
    for attr in ("runs", "epochs", "device", "out_dir"):
        value = getattr(args, attr)
        if value is not None:
            setattr(cfg, attr, value)
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if not torch.cuda.is_available():
        cfg.device = "cpu"

    tasks = D.all_transfer_tasks()
    if args.tasks:
        wanted = {t.replace("2", "").upper() for t in args.tasks}
        tasks = [(s, t) for s, t in tasks if f"{s}{t}" in wanted]

    names = args.methods if args.methods else \
        list(classic.CLASSIC_METHODS) + list(DEEP_METHODS)

    os.makedirs(cfg.out_dir, exist_ok=True)
    rows = []

    for name in names:
        accuracies = {}
        for source, target in tasks:
            scores = []
            for run in range(cfg.runs):
                seed = cfg.seed + run
                if name in classic.CLASSIC_METHODS:
                    scores.append(run_classic(name, cfg, source, target, seed))
                else:
                    scores.append(run_deep(DEEP_METHODS[name], cfg, source, target, seed))
            accuracies[f"{source}->{target}"] = float(np.mean(scores))
        average = float(np.mean(list(accuracies.values())))
        tag = "  (proposed)" if name == PROPOSED else ""
        detail = "  ".join(f"{k} {v:6.2f}" for k, v in accuracies.items())
        print(f"  {name:<10} {detail}   Average {average:6.2f}{tag}")
        rows.append({"method": name, "runs": cfg.runs,
                     **{k: round(v, 2) for k, v in accuracies.items()},
                     "Average": round(average, 2)})

    csv_path = os.path.join(cfg.out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(cfg.out_dir, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
