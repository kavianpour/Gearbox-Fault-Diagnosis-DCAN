# Method — Deep Coral Adversarial Network (DCAN)

This document describes the DCAN method in full: the network structure, the three loss functions and their optimization, and the training configuration reported in the paper.

---

## 1. Overview

DCAN is an end-to-end model for **unsupervised domain adaptation (UDA)** in gearbox fault diagnosis. It is trained on a *labeled* source operating condition and an *unlabeled* target operating condition, and learns features that are both discriminative for fault type and invariant to the operating condition.

The architecture is split into two parts:

1. **Condition identification** — a 1-D CNN feature extractor + a gearbox health-status classifier.
2. **Domain adaptation** — two modules that reduce the source/target distribution gap: an **adversarial domain discriminator** and a **deep CORAL** alignment.

---

## 2. Condition identification (feature extractor + classifier)

The feature extractor is a one-dimensional CNN with **four convolution layers**, each followed by a **max-pooling layer**, then **two fully-connected layers** and an output layer.

Design choices reported in the paper:

- **Input:** raw 1-D vibration signal, normalized with **z-score**. The batch mixes labeled source samples and unlabeled target samples.
- **Depth:** four convolution layers increase representational depth so the network can learn from complex, nonlinear signal structure.
- **Max-pooling** after each convolution reduces dimensionality, parameter count, and overfitting.
- **ReLU** activations speed up training and mitigate exploding/vanishing gradients.
- **Batch normalization** after the first convolution reduces internal covariate shift and improves generalization and training speed.
- **Dropout (rate 0.5)** after the fourth max-pooling layer to further limit overfitting.
- **Padding** is applied in every convolution layer.
- The network ends with a **flatten** layer, several **FC** layers, and a **softmax** output.
- **Pseudo-label learning** labels the target domain: each target sample takes the class with the highest predicted probability.

### Architecture table (Table I in the paper)

| Section | Symbol | Size / stride / #filters | Output |
|---|---|---|---|
| Condition identification | C1 | 32 / 8 / 32 | 253 × 32 |
| | P1 | 2 / 2 / 32 | 126 × 32 |
| | C2 | 3 / 2 / 64 | 124 × 64 |
| | P2 | 2 / 2 / 64 | 62 × 64 |
| | C3 | 3 / 2 / 64 | 60 × 64 |
| | P3 | 2 / 2 / 64 | 30 × 64 |
| | C4 | 3 / 1 / 64 | 28 × 64 |
| | P4 | 2 / 2 / 64 | 14 × 64 |
| | Flatten | — | 896 × 1 |
| Domain adaptation | FC1 | 256 | 256 × 1 |
| | FC2 | 128 | 128 × 1 |
| | FC3 | 128 | 128 × 1 |
| | FC4 | 128 | 128 × 1 |
| | FC5 | 5 | 5 × 1 |

> Note: the layer table is transcribed directly from the paper. Its own "size / stride / filters" column lists C2 and C3 with stride 2, but that is inconsistent with the "output" column on the same row — those output sizes (124 and 60) are reproduced only if C2 and C3 use stride **1**; stride 2 gives 62 and 30 instead. The arithmetic is checked layer by layer in `config.py`. This implementation uses stride 1 for C2 and C3, which is what makes the flatten size come out to 896 as the table states.

---

## 3. Optimization objective

The total objective combines three losses:

$$\mathcal{L} = \mathcal{L}_c + \alpha\,\mathcal{L}_d + \beta\,\mathcal{L}_{coral}$$

where $\alpha$ and $\beta$ control how strongly each adaptation module influences training.

### 3.1 Classification loss $\mathcal{L}_c$ (Object 1)

Softmax/cross-entropy on the **labeled source domain**, minimizing health-condition classification error:

$$\mathcal{L}_c = \frac{1}{n_s}\left[ \sum_{i=1}^{b}\sum_{j=1}^{K} \mathbb{I}[y_i = C]\,\log \frac{\exp((w_j)^T f + b)}{\sum_{m=1}^{K}\exp((w_m)^T f + b)} \right]$$

where $n_s$ is the number of source samples, $K$ is the number of health classes, and $\mathbb{I}[\cdot]$ is the indicator function.

### 3.2 CORAL loss $\mathcal{L}_{coral}$ (Object 2)

Where MMD-based methods only align **first-order** statistics, CORAL aligns **second-order** statistics by minimizing the distance between source and target covariance matrices:

$$\mathcal{L}_{coral} = \frac{1}{4d^2}\,\lVert C_s - C_t \rVert_F^2$$

with $d$ the feature-space dimension and $\lVert\cdot\rVert_F^2$ the squared Frobenius norm. The covariance matrices are

$$C_s = \frac{1}{n_s - 1}\left(\chi_s^T \chi_s - \frac{1}{n_s}(\mathbf{1}^T \chi_s)^T(\mathbf{1}^T \chi_s)\right)$$

$$C_t = \frac{1}{n_t - 1}\left(\chi_t^T \chi_t - \frac{1}{n_t}(\mathbf{1}^T \chi_t)^T(\mathbf{1}^T \chi_t)\right)$$

where $\mathbf{1}^T$ is an all-ones row vector and $\chi$ corresponds to the FC2 features.

### 3.3 Adversarial discriminator loss $\mathcal{L}_d$ (Object 3)

A two-player minimax game between feature extractor $F$ and domain discriminator $D$. $D$ learns to separate source from target features; $F$ learns to fool $D$, so the resulting features are domain-invariant. Using binary cross-entropy:

$$\mathcal{L}_d(x_s, x_t) = \frac{-1}{n_s + n_t}\Big( \mathbb{E}_{x_s \in D_s}[\log D(F(x_s))] + \mathbb{E}_{x_t \in D_t}[\log(1 - D(F(x_t)))] \Big)$$

where $\mathbb{E}$ is the expectation and $x_s, x_t$ are source and target samples.

---

## 4. Parameter optimization

Training uses **stochastic gradient descent** to solve

$$\mathcal{L}(\theta_f^*, \theta_c^*, \theta_d^*) = \min_{\theta_f, \theta_c, \theta_d}\big( \mathcal{L}_c(\theta_f, \theta_c) + \alpha\,\mathcal{L}_d(\theta_f, \theta_d) + \beta\,\mathcal{L}_{coral}(\theta_f) \big)$$

with parameter updates (learning rate $\gamma$):

$$\theta_f \leftarrow \theta_f - \gamma\left(\frac{\partial \mathcal{L}_c}{\partial \theta_f} + \alpha\frac{\partial \mathcal{L}_d}{\partial \theta_f} + \beta\frac{\partial \mathcal{L}_{coral}}{\partial \theta_f}\right)$$

$$\theta_c \leftarrow \theta_c - \gamma\,\frac{\partial \mathcal{L}_c}{\partial \theta_c} \qquad\qquad \theta_d \leftarrow \theta_d - \gamma\,\frac{\partial \mathcal{L}_d}{\partial \theta_d}$$

Here $\theta_f, \theta_c, \theta_d$ are the parameters of the feature extractor, the health-status classifier, and the domain discriminator, respectively.

---

## 5. Training setup (implementation details)

| Setting | Value |
|---|---|
| Weight initialization | Xavier |
| Batch size | **128** (selected from {16, 32, 64, 128, 256}) |
| Initial learning rate | **0.001**, decayed as epochs increase |
| Trade-off $\alpha$ (adversarial) | **1** |
| Trade-off $\beta$ (CORAL) | **0.5** |
| Optimizer | Stochastic gradient descent |
| Repetitions | Each experiment repeated **10×**, results averaged |

The $\alpha,\beta$ values were chosen by sweeping {0.01, 0.1, 0.5, 1}; the best task-A→B accuracy (93.6 %) occurs at $\alpha = 1, \beta = 0.5$, while $\alpha = \beta = 0$ reduces DCAN to a plain CNN (80.23 %). See the sensitivity figure in the [README](../README.md#two-key-ideas).

---

## 6. Where each piece lives in the code

| Component | File | Entry point |
|---|---|---|
| Hyper-parameters | [`config.py`](../config.py) | `Config` dataclass |
| SEU loader, windowing, z-score, transfer tasks | [`data.py`](../data.py) | `build_condition_dataset`, `transfer_loaders` |
| Condition-identification CNN (F) | [`model.py`](../model.py) | `FeatureExtractor` |
| Gradient reversal layer | [`model.py`](../model.py) | `grad_reverse`, `calc_coeff` |
| Domain discriminator (D) | [`model.py`](../model.py) | `DomainDiscriminator` |
| Full network | [`model.py`](../model.py) | `DCAN` |
| CORAL, MMD, fixed-bandwidth MKMMD | [`da_losses.py`](../da_losses.py) | `coral_loss`, `mmd_loss`, `mkmmd_loss` |
| Method registry (CNN / C-MKMMD / C-CORAL / DCAN) | [`methods.py`](../methods.py) | `DEEP_METHODS` |
| JDA and the shallow DANN baseline | [`classic.py`](../classic.py) | `run_jda`, `ShallowDANN` |
| Training loop, Eq. 6 objective | [`train.py`](../train.py) | `run_deep` |
| Original explanatory figures | [`make_figures.py`](../make_figures.py) | `adaptation_concept`, `transfer_tasks` |

---

*Equations and tables in this document are transcribed from the published paper for documentation purposes. Figures remain © 2022 IEEE — see [LICENSE](../LICENSE).*
