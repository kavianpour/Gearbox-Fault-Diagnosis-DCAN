"""
DCAN model (Section II of the paper; Fig. 1).

Pipeline for a batch of N samples:

    raw vibration (N, 1, 2048)
      -> C1  32/8/32  + BN + ReLU -> P1  2/2      (Table I)
      -> C2  3/1/64        + ReLU -> P2  2/2
      -> C3  3/1/64        + ReLU -> P3  2/2
      -> C4  3/1/64        + ReLU -> P4  2/2 -> dropout 0.5
      -> flatten  896
      -> FC1  256
      -> FC2  128    (Fig. 1 labels this FC2S for a source batch, FC2T for a
                       target batch - it is one shared layer, applied to
                       whichever domain the current batch is drawn from)
    Heads (Fig. 1):
      classifier          : FC5 -> 5 classes + softmax      (Eq. 1)
                             (Fig. 1: FC5S for source; FC5T is the same head
                             applied to target features, read out for pseudo-
                             labelling rather than for the classification loss)
      domain discriminator: GRL -> FC3 128 -> FC4 128 -> 1  (Eq. 5)
                             takes both FC2S and FC2T as input
      CORAL               : second-order statistics of FC2S and FC2T (Eq. 2-4)
                             (the bidirectional "CORAL" arrow between them in
                             Fig. 1)

Batch normalisation is applied after the first convolution only, as the paper
describes. Convolutions are unpadded; see the note on `cnn_strides` in
`config.py` for how the stride of C2/C3 was determined from Table I.

A gradient reversal layer implements the two-player minimax game of Eq. 5:
the discriminator minimises its binary cross-entropy while the feature
extractor receives the negated gradient and therefore maximises it.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


# --------------------------------------------------------------------------- #
# Gradient Reversal Layer (Eq. 5, Eq. 8-10)
# --------------------------------------------------------------------------- #
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
                 - (high - low) + low)


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coeff * grad_output, None


def grad_reverse(x, coeff=1.0):
    return _GradReverse.apply(x, coeff)


# --------------------------------------------------------------------------- #
# Condition identification: feature extractor F(.)  (Section II.A.1)
# --------------------------------------------------------------------------- #
class FeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []
        in_c = 1
        for i, (c, k, s) in enumerate(zip(cfg.cnn_channels, cfg.cnn_kernels,
                                          cfg.cnn_strides)):
            layers.append(nn.Conv1d(in_c, c, kernel_size=k, stride=s))
            if i == 0:
                layers.append(nn.BatchNorm1d(c))     # BN after the first convolution
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=cfg.pool_size, stride=cfg.pool_stride))
            in_c = c
        layers.append(nn.Dropout(cfg.dropout))       # after the fourth pooling layer
        self.cnn = nn.Sequential(*layers)

        with torch.no_grad():
            flat_dim = self.cnn(torch.zeros(1, 1, cfg.sample_length)).numel()
        self.flat_dim = flat_dim

        self.fc1 = nn.Sequential(nn.Linear(flat_dim, cfg.fc1_dim), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(cfg.fc1_dim, cfg.fc2_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        h = self.cnn(x)
        h = h.reshape(h.size(0), -1)
        return self.fc2(self.fc1(h))                # (N, 128) - the CORAL features


# --------------------------------------------------------------------------- #
# Domain discriminator D(.)  (Section II.A.2, Eq. 5)
# --------------------------------------------------------------------------- #
class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, coeff):
        x = grad_reverse(x, coeff)
        return torch.sigmoid(self.net(x)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Full DCAN model
# --------------------------------------------------------------------------- #
class DCAN(nn.Module):
    """
    Configurable backbone shared by all deep methods. `spec` (methods.MethodSpec)
    toggles the adversarial discriminator and selects the distribution-matching
    loss. With the default spec it is the proposed DCAN.
    """

    def __init__(self, cfg, spec=None, num_classes=None):
        super().__init__()
        self.use_adversarial = True if spec is None else spec.use_adversarial
        n_cls = num_classes if num_classes is not None else cfg.num_classes

        self.F = FeatureExtractor(cfg)
        self.classifier = nn.Linear(cfg.fc2_dim, n_cls)                 # FC5 (Eq. 1)
        self.discriminator = DomainDiscriminator(cfg.fc2_dim, cfg.disc_hidden)

    def forward(self, x):
        feat = self.F(x)
        return feat, self.classifier(feat)


def xavier_init(module):
    """Xavier initialisation (Section III.B)."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
