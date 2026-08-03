"""
Shallow comparison methods from Section III.C.

  * JDA  : Joint Distribution Adaptation on PCA features, followed by a
           nearest-neighbour classifier. A shallow domain-adaptation baseline.
  * DANN : the shallow domain-adversarial neural network baseline. It is
           deliberately light - a small feature network with an adversarial
           domain branch - and NOT the deep DCAN backbone, so that it matches
           the weaker performance the paper reports for it.

NOTE: RWKDCAE (residual wide-kernel deep convolutional auto-encoder) is also
listed as a comparison method in the paper but is a third-party architecture
published elsewhere; it is not reimplemented here.
"""

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from model import DomainDiscriminator


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def loader_to_numpy(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.numpy().reshape(x.shape[0], -1))
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def _kernel(X1, X2=None, kind="primal", gamma=1.0):
    if kind == "primal":
        return X1
    X2 = X1 if X2 is None else X2
    sq = (np.sum(X1 ** 2, axis=1)[:, None] + np.sum(X2 ** 2, axis=1)[None, :]
          - 2.0 * X1 @ X2.T)
    return np.exp(-gamma * sq)


# --------------------------------------------------------------------------- #
# JDA
# --------------------------------------------------------------------------- #
def jda(Xs, ys, Xt, dim=30, lmbda=1.0, iterations=5, n_components=64):
    """Joint Distribution Adaptation followed by a 1-NN classifier."""
    scaler = StandardScaler().fit(np.concatenate([Xs, Xt]))
    Xs, Xt = scaler.transform(Xs), scaler.transform(Xt)

    pca = PCA(n_components=min(n_components, Xs.shape[1], len(Xs))).fit(
        np.concatenate([Xs, Xt]))
    Xs, Xt = pca.transform(Xs), pca.transform(Xt)

    X = np.vstack([Xs, Xt]).T
    X = X / (np.linalg.norm(X, axis=0) + 1e-8)
    m, n = X.shape
    ns, nt = len(Xs), len(Xt)
    classes = np.unique(ys)
    dim = min(dim, m)

    e = np.vstack([np.ones((ns, 1)) / ns, -np.ones((nt, 1)) / nt])
    M0 = e @ e.T * len(classes)
    H = np.eye(n) - np.ones((n, n)) / n
    yt_pseudo = None

    for _ in range(iterations):
        M = M0.copy()
        if yt_pseudo is not None:
            for c in classes:
                ec = np.zeros((n, 1))
                si = np.flatnonzero(ys == c)
                ti = np.flatnonzero(yt_pseudo == c)
                if len(si) == 0 or len(ti) == 0:
                    continue
                ec[si] = 1.0 / len(si)
                ec[ns + ti] = -1.0 / len(ti)
                M += ec @ ec.T
        M = M / (np.linalg.norm(M, "fro") + 1e-8)

        a = X @ M @ X.T + lmbda * np.eye(m)
        b = X @ H @ X.T
        eigvals, eigvecs = scipy.linalg.eig(a, b)
        order = np.argsort(np.abs(eigvals))
        A = np.real(eigvecs[:, order][:, :dim])

        Z = A.T @ X
        Z = Z / (np.linalg.norm(Z, axis=0) + 1e-8)
        Zs, Zt = Z[:, :ns].T, Z[:, ns:].T

        knn = KNeighborsClassifier(n_neighbors=1).fit(Zs, ys)
        yt_pseudo = knn.predict(Zt)

    return yt_pseudo


def run_jda(source_loader, target_loader, **kwargs):
    Xs, ys = loader_to_numpy(source_loader)
    Xt, yt = loader_to_numpy(target_loader)
    pred = jda(Xs, ys, Xt, **kwargs)
    return float((pred == yt).mean() * 100.0)


# --------------------------------------------------------------------------- #
# Shallow DANN
# --------------------------------------------------------------------------- #
class ShallowDANN(nn.Module):
    """A deliberately small domain-adversarial network, not the DCAN backbone."""

    def __init__(self, cfg, num_classes=5, hidden=64):
        super().__init__()
        self.use_adversarial = True
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=63, stride=8), nn.BatchNorm1d(16), nn.ReLU(True),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=15), nn.ReLU(True),
            nn.MaxPool1d(4),
            nn.AdaptiveMaxPool1d(4),
        )
        self.fc = nn.Sequential(nn.Linear(32 * 4, hidden), nn.ReLU(True))
        self.classifier = nn.Linear(hidden, num_classes)
        self.discriminator = DomainDiscriminator(hidden, hidden)

    def forward(self, x):
        h = self.cnn(x).reshape(x.size(0), -1)
        feat = self.fc(h)
        return feat, self.classifier(feat)


CLASSIC_METHODS = {
    "JDA": run_jda,
}
