"""
Distribution-matching losses used by DCAN and its comparison methods.

  * CORAL : correlation alignment (Eq. 2-4). Matches the SECOND-order
            statistics of the two domains by minimising the Frobenius distance
            between their feature covariance matrices. This is the loss the
            proposed method uses, and the paper's motivation for it is that
            MMD-style criteria only align first-order statistics.
  * MMD   : multi-kernel maximum mean discrepancy with adaptive bandwidths.
  * MKMMD : multi-kernel MMD with the FIVE fixed bandwidths the paper specifies
            for the C-MKMMD comparison, {0.001, 0.01, 1, 10, 100}.

All three are computed on the 128-dimensional FC2 features.
"""

import torch

MKMMD_BANDWIDTHS = [0.001, 0.01, 1.0, 10.0, 100.0]


def _covariance(x):
    """Covariance matrix of a feature batch (Eq. 3-4)."""
    n = x.size(0)
    ones = torch.ones(1, n, device=x.device, dtype=x.dtype)
    mean = ones @ x                                   # (1, d)
    return (x.t() @ x - (mean.t() @ mean) / n) / (n - 1)


def coral_loss(source, target):
    """Eq. 2: ||C_s - C_t||_F^2 / (4 d^2)."""
    d = source.size(1)
    diff = _covariance(source) - _covariance(target)
    return (diff * diff).sum() / (4.0 * d * d)


def _pairwise_sq_dists(source, target):
    total = torch.cat([source, target], dim=0)
    diff = total.unsqueeze(0) - total.unsqueeze(1)
    return (diff ** 2).sum(dim=2)


def _kernel_matrix(source, target, bandwidths=None, kernel_mul=2.0, kernel_num=5):
    l2 = _pairwise_sq_dists(source, target)
    if bandwidths is None:
        n = source.size(0) + target.size(0)
        base = l2.detach().sum() / (n * n - n + 1e-8)
        base = base / (kernel_mul ** (kernel_num // 2))
        bandwidths = [base * (kernel_mul ** i) for i in range(kernel_num)]
    return sum(torch.exp(-l2 / (bw + 1e-8)) for bw in bandwidths)


def _mmd_from_kernel(kernels, batch_size):
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return (xx + yy - xy - yx).mean()


def mmd_loss(source, target):
    """Multi-kernel MMD with bandwidths estimated from the batch."""
    return _mmd_from_kernel(_kernel_matrix(source, target), source.size(0))


def mkmmd_loss(source, target, bandwidths=None):
    """Multi-kernel MMD with the five fixed bandwidths of the C-MKMMD method."""
    bandwidths = MKMMD_BANDWIDTHS if bandwidths is None else bandwidths
    kernels = _kernel_matrix(source, target, bandwidths=bandwidths)
    return _mmd_from_kernel(kernels, source.size(0))


DA_LOSSES = {
    "coral": coral_loss,
    "mmd": mmd_loss,
    "mkmmd": mkmmd_loss,
}


def get_da_loss(name):
    if name in (None, "none"):
        return None
    if name not in DA_LOSSES:
        raise ValueError(f"unknown distribution-matching loss '{name}'")
    return DA_LOSSES[name]
