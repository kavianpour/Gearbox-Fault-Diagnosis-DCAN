"""
Central configuration for the DCAN implementation.

DCAN = Deep CORAL Adversarial Network
Reference: M. Kavianpour, M. Ghorvei, P. Kavianpour, A. Ramezani,
M. T. H. Beheshti, "An Intelligent Gearbox Fault Diagnosis under Different
Operating Conditions using Adversarial Domain Adaptation",
2022 8th International Conference on Control, Instrumentation and Automation
(ICCIA), IEEE, 2022.

Layer sizes follow Table I ("Structural details of the proposed DCAN method");
the training settings follow Section III.B ("Implementation details").
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ----- data (Section III.A) -----
    data_root: str = "./SEU"         # directory holding the SEU gearbox CSV files
    channel: int = 1                 # second of the eight vibration channels (0-based)
    sample_length: int = 2048        # window length implied by the Table I output sizes
    window_step: int = 1024          # sliding-window step (50% overlap)
    n_samples_per_class: int = 1000  # windows kept per class per operating condition
    train_ratio: float = 0.8         # 80% train / 20% test
    num_classes: int = 5             # Health, Chipped, Root, Miss, Surface (Table II)
    normalize: str = "z-score"       # raw signals normalised with the z-score method

    # ----- model (Table I) -----
    #   C1 32/8/32 -> P1 -> C2 3/1/64 -> P2 -> C3 3/1/64 -> P3 -> C4 3/1/64 -> P4
    #
    # Table I lists C2 and C3 with stride 2 in its "size / stride / filters"
    # column, but its own "output" column is only consistent with stride 1:
    #   C1  (32, s=8) on 2048         -> 253   (matches, stride 8)
    #   P1  (2,  s=2) on 253          -> 126
    #   C2  (3,  s=1) on 126          -> 124   (matches the table's 124)
    #       C2  (3,  s=2) on 126      ->  62   (does NOT match)
    #   P2  (2,  s=2) on 124          ->  62
    #   C3  (3,  s=1) on 62           ->  60   (matches the table's 60)
    #       C3  (3,  s=2) on 62       ->  30   (does NOT match)
    #   P3  (2,  s=2) on 60           ->  30
    #   C4  (3,  s=1) on 30           ->  28   (matches; C4 is already stride 1)
    #   P4  (2,  s=2) on 28           ->  14
    #   flatten: 14 * 64 = 896        (matches the table's 896)
    # so every layer after C1 is stride 1; the "2" entries for C2/C3 are taken
    # to be a transcription error in the table rather than the intended value.
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 64, 64])
    cnn_kernels: List[int] = field(default_factory=lambda: [32, 3, 3, 3])
    cnn_strides: List[int] = field(default_factory=lambda: [8, 1, 1, 1])
    pool_size: int = 2
    pool_stride: int = 2
    dropout: float = 0.5             # after the fourth max-pooling layer
    fc1_dim: int = 256               # FC1
    fc2_dim: int = 128               # FC2 - the layer the CORAL loss is computed on
    disc_hidden: int = 128           # FC3 / FC4 of the domain discriminator

    # ----- optimisation (Section III.B) -----
    lr: float = 1e-3                 # initial learning rate, decayed over epochs
    lr_gamma: float = 0.98           # exponential decay applied per epoch
    momentum: float = 0.9            # stochastic gradient descent with momentum
    weight_decay: float = 5e-4
    batch_size: int = 128
    epochs: int = 100
    alpha: float = 1.0               # trade-off of the adversarial loss   (Eq. 6)
    beta: float = 0.5                # trade-off of the CORAL loss         (Eq. 6)
    grl_alpha: float = 10.0          # gradient-reversal ramp-up steepness
    grl_max_iter: float = 10000.0    # gradient-reversal ramp-up horizon
    warmup_epochs: int = 1           # source-only epochs before adaptation starts

    # ----- experiment -----
    runs: int = 10                   # each experiment repeated 10 times, averaged
    seed: int = 0
    device: str = "cuda"             # "cuda" | "cpu"
    out_dir: str = "./results"


cfg = Config()
