"""
Southeast University (SEU) gearbox dataset loader (Section III.A of the paper).

Setup used in the paper:
  * gearbox sub-dataset of the Drivetrain Dynamics Simulator recordings,
    second of the eight vibration channels
  * 5 health states (Table II):
      C0 Health   - normal condition
      C1 Chipped  - crack in the gear feet
      C2 Root     - crack in the root of the gear feet
      C3 Miss     - missing feet in the gear
      C4 Surface  - wear on the gear surface
  * 2 operating conditions -> transfer tasks:
      A = 20 Hz - 0 V
      B = 30 Hz - 2 V
  * raw vibration normalised with the z-score method
  * 80% of the data for training, 20% for testing
  * the source domain is labelled, the target domain is unlabelled

Download: https://github.com/cathysiyu/Mechanical-datasets
Place the gearset CSV files (Health_20_0.csv, Chipped_20_0.csv, ...,
Surface_30_2.csv) directly under `cfg.data_root`.

The CSV files are not uniformly delimited - most are tab separated while some
are comma separated, and all carry a short instrument header - so the reader
sniffs the delimiter and skips any line that does not parse as numbers.
"""

import os
import glob

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# class label -> filename prefix in the Mechanical-datasets repository
SEU_PREFIX = {0: "Health", 1: "Chipped", 2: "Root", 3: "Miss", 4: "Surface"}
SEU_CLASS_ORDER = ["Health", "Chipped", "Root", "Miss", "Surface"]

# operating condition -> filename suffix (rotating speed Hz, load V)
SEU_CONDITIONS = {"A": "20_0", "B": "30_2"}
SEU_COND_ORDER = ["A", "B"]


def _find_csv(root, prefix, suffix):
    for pattern in (f"{prefix}_{suffix}.csv", f"{prefix}_{suffix}*.csv",
                    f"{prefix.lower()}_{suffix}*.csv"):
        hits = sorted(glob.glob(os.path.join(root, pattern)))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No SEU file for '{prefix}' at condition '{suffix}' in {root}")


def _read_channel(path, channel):
    """Read one vibration channel, tolerating mixed delimiters and headers."""
    values = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            parts = line.replace(",", "\t").split("\t")
            if len(parts) <= channel:
                continue
            try:
                values.append(float(parts[channel]))
            except ValueError:
                continue          # instrument header line
    if not values:
        raise ValueError(f"No numeric data in channel {channel} of {path}")
    return np.asarray(values, dtype=np.float32)


def _normalize(window, method):
    if method == "z-score":
        return (window - window.mean()) / (window.std() + 1e-8)
    if method == "0-1":
        span = window.max() - window.min()
        return (window - window.min()) / (span + 1e-8)
    return window


def _windows(signal, length, step, n_max, method):
    out, idx = [], 0
    while idx + length <= len(signal) and len(out) < n_max:
        out.append(_normalize(signal[idx:idx + length], method))
        idx += step
    return np.stack(out, axis=0).astype(np.float32)


def build_condition_dataset(cfg, condition, rng):
    """Train / test TensorDatasets for one operating condition."""
    suffix = SEU_CONDITIONS[condition]
    X_tr, y_tr, X_te, y_te = [], [], [], []

    for label, prefix in SEU_PREFIX.items():
        signal = _read_channel(_find_csv(cfg.data_root, prefix, suffix), cfg.channel)
        windows = _windows(signal, cfg.sample_length, cfg.window_step,
                           cfg.n_samples_per_class, cfg.normalize)
        windows = windows[rng.permutation(len(windows))]
        n_train = int(round(len(windows) * cfg.train_ratio))
        X_tr.append(windows[:n_train])
        y_tr.append(np.full(n_train, label))
        X_te.append(windows[n_train:])
        y_te.append(np.full(len(windows) - n_train, label))

    def _dataset(X, y):
        X = np.concatenate(X, axis=0)[:, None, :]
        y = np.concatenate(y, axis=0)
        return TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())

    return _dataset(X_tr, y_tr), _dataset(X_te, y_te)


def transfer_loaders(cfg, source, target, seed):
    """Loaders for one transfer task: labelled source, unlabelled target."""
    rng = np.random.default_rng(seed)
    src_train, src_test = build_condition_dataset(cfg, source, rng)
    tgt_train, tgt_test = build_condition_dataset(cfg, target, rng)
    bs = cfg.batch_size
    return {
        "source_train": DataLoader(src_train, batch_size=bs, shuffle=True, drop_last=True),
        "source_test": DataLoader(src_test, batch_size=bs, shuffle=False),
        "target_train": DataLoader(tgt_train, batch_size=bs, shuffle=True, drop_last=True),
        "target_test": DataLoader(tgt_test, batch_size=bs, shuffle=False),
    }


def all_transfer_tasks():
    """The two cross-condition tasks: A -> B and B -> A."""
    return [("A", "B"), ("B", "A")]
