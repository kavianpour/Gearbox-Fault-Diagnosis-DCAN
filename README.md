# Intelligent Gearbox Fault Diagnosis under Different Operating Conditions via Adversarial Domain Adaptation

> A hybrid unsupervised domain-adaptation framework (**DCAN** — Deep Coral Adversarial Network) that diagnoses gearbox faults when the operating speed and load shift between training and deployment, and when the target data is completely unlabeled.

<p align="left">
  <a href="https://doi.org/10.1109/ICCIA54998.2022.9737160"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1109%2FICCIA54998.2022.9737160-1f7a8c"></a>
  <a href="https://ieeexplore.ieee.org/document/9737160"><img alt="Venue" src="https://img.shields.io/badge/Venue-IEEE%20ICCIA%202022-00488d"></a>
  <img alt="Year" src="https://img.shields.io/badge/Year-2022-555">
  <img alt="Citations" src="https://img.shields.io/badge/citations-17%2B-2e8b57">
  <img alt="Docs License" src="https://img.shields.io/badge/Docs%20License-CC%20BY%204.0-lightgrey">
  <img alt="Status" src="https://img.shields.io/badge/code-not%20yet%20public-orange">
</p>

**Authors:** [Mohammadreza Kavianpour](https://github.com/kavianpour)¹, Mohammadreza Ghorvei¹, Parisa Kavianpour², Amin Ramezani¹ \*, Mohammad T. H. Beheshti¹

¹ Department of Electrical and Computer Engineering, *Tarbiat Modares University*, Tehran, Iran
² *University of Mazandaran*, Babolsar, Iran
\* Corresponding author

Published in the **2022 8th International Conference on Control, Instrumentation and Automation (ICCIA)**, IEEE.
DOI: [10.1109/ICCIA54998.2022.9737160](https://doi.org/10.1109/ICCIA54998.2022.9737160)

---

## TL;DR

Deep-learning fault classifiers collapse when the machine they were trained on starts running at a different speed or load — the data distribution shifts, and in the real world the new condition has **no labels**. This paper proposes **DCAN**, a one-dimensional CNN coupled with *two* complementary domain-adaptation modules (deep CORAL + an adversarial domain discriminator) that align a labeled *source* operating condition with an unlabeled *target* operating condition. On the SEU gearbox dataset it reaches **91.94 % average accuracy** across two cross-condition transfer tasks, beating every shallow and deep baseline tested.

The three core ideas:

1. **Hybrid (not single) domain matching.** Combining a *second-order statistical* alignment (CORAL) with an *adversarial* alignment extracts more domain-invariant features than either module alone.
2. **Fully unsupervised target.** The source domain is labeled; the target domain is unlabeled and handled with pseudo-label learning, matching the realistic industrial setting.
3. **End-to-end raw-signal pipeline.** Raw z-score-normalized vibration signals go straight into a 1-D CNN — no hand-crafted feature engineering.

> [!NOTE]
> **Documentation & resources only.** This repository currently contains **documentation, the dataset description, and figures extracted from the paper** — it is a curated showcase of the published work. The training/inference source code is **not yet public** (see the [Roadmap](#roadmap)). If you need details beyond what is documented here, please reach out or cite the paper.

---

## The framework / architecture

DCAN has two cooperating halves. The **condition-identification** half is a 1-D CNN feature extractor plus a gearbox health classifier. The **domain-adaptation** half reduces the distribution gap between the source and target operating conditions using two modules working on the fully-connected features: an **adversarial domain discriminator** and a **deep CORAL** alignment.

![DCAN architecture](assets/dcan_architecture.png)

*Architecture of the proposed DCAN method (Fig. 1 in the paper). Forward propagation flows left→right through the CNN; three back-propagation paths train the classifier ($L_c$), the domain discriminator ($L_d$), and the CORAL alignment ($L_{coral}$).* © 2022 IEEE — see [LICENSE](LICENSE).

The total objective combines three losses:

$$\mathcal{L} = \mathcal{L}_c + \alpha\,\mathcal{L}_d + \beta\,\mathcal{L}_{coral}$$

where $\mathcal{L}_c$ is the source classification (softmax) loss, $\mathcal{L}_d$ is the adversarial discriminator (binary cross-entropy) loss, $\mathcal{L}_{coral}$ is the second-order CORAL loss, and $\alpha,\beta$ weight each adaptation module. Full equations and the layer table are in [`docs/method.md`](docs/method.md).

---

## Why this is hard

| Challenge | Why it breaks ordinary models | How DCAN responds |
|---|---|---|
| **Shifting operating conditions** | Changing speed/load makes vibration data non-stationary and shifts its distribution, so a model trained on one condition mispredicts on another. | Aligns source and target distributions in latent space instead of assuming they match. |
| **No labels in the target domain** | Real deployments cannot label faults on the live machine; supervised fine-tuning is impossible. | Unsupervised domain adaptation + pseudo-labeling — the target needs no labels. |
| **First-order alignment isn't enough** | MMD-style methods only match means (first-order statistics), leaving correlation structure misaligned. | CORAL matches **second-order** statistics (covariance), capturing feature correlations. |
| **A single alignment module leaves a residual gap** | One matching criterion captures only part of the discrepancy. | Two modules (statistical + adversarial) jointly squeeze out more domain-invariant features. |
| **Hand-crafted features don't transfer** | Features tuned for one regime are unreliable in another and need expert tuning. | End-to-end 1-D CNN learns features directly from the raw signal. |

---

## Two key ideas

**1. Deep CORAL — align the *correlation*, not just the mean.** CORAL minimizes the distance between the covariance matrices of the source and target features, so the model matches second-order statistics that mean-based methods (MMD) miss. This is what lets the learned features stay valid across operating conditions.

**2. Adversarial domain discriminator — a two-player game.** A discriminator $D$ is trained to tell source features from target features, while the feature extractor $F$ is trained to *fool* it. At equilibrium, $F$ produces features that $D$ can no longer attribute to either domain — i.e., domain-invariant representations. Stacking this on top of CORAL is the "hybrid" in DCAN.

![Sensitivity to the alpha and beta trade-off coefficients](assets/sensitivity_alpha_beta.png)

*Accuracy on task A → B as the trade-off coefficients $\alpha$ (adversarial) and $\beta$ (CORAL) vary (Fig. 2 in the paper). The peak — 93.6 % — occurs at $\alpha = 1,\ \beta = 0.5$; with $\alpha = \beta = 0$ the model degrades to a plain CNN (80.23 %).* © 2022 IEEE.

---

## Headline results

On the SEU gearbox dataset with two operating conditions — **task A** (20 Hz, 0 V) and **task B** (30 Hz, 2 V) — DCAN is evaluated on both transfer directions (A → B and B → A) and compared against shallow domain adaptation (JDA, DANN), a deep baseline without adaptation (RWKDCAE, CNN), and single-module variants (C-MKMMD, C-CORAL).

![Results table](assets/results_table3.png)

*Accuracy (%) under changing working conditions (Table III in the paper). © 2022 IEEE.*

Highlights:

- **91.94 % average accuracy** — the best of all methods compared.
- **+13.07 %** over the plain CNN baseline and **+9.98 %** over RWKDCAE, showing the value of domain adaptation itself.
- **+3.49 %** over the best single-module variant (C-CORAL, 88.45 %), showing the value of the *hybrid* design.
- C-CORAL beats C-MKMMD, confirming that second-order (CORAL) alignment outperforms MMD here.

> Each experiment was repeated 10 times and averaged to reduce randomness.

---

## Dataset

| | |
|---|---|
| **Name** | SEU gearbox dataset (Southeast University, China) |
| **Acquisition** | Drivetrain Dynamics Simulator (DDS), 8 vibration channels |
| **Channel used** | Channel 2 (gearbox sub-dataset) |
| **Operating conditions** | Task A: 20 Hz – 0 V · Task B: 30 Hz – 2 V |
| **Health classes (5)** | C0 Health · C1 Chipped · C2 Root · C3 Miss · C4 Surface |
| **Split** | 80 % train / 20 % test; source labeled, target unlabeled |
| **Preprocessing** | z-score normalization of the raw 1-D vibration signal |

Full class definitions, task design, and download pointers are in [`docs/datasets.md`](docs/datasets.md).

---

## Repository contents

```
Gearbox-Fault-Diagnosis-DCAN/
├── README.md                 ← you are here
├── assets/                   ← key figures extracted from the paper (© IEEE)
│   ├── dcan_architecture.png
│   ├── sensitivity_alpha_beta.png
│   └── results_table3.png
├── docs/
│   ├── method.md             ← full method, architecture table, training setup
│   ├── challenges.md         ← the real-world problems this work targets
│   └── datasets.md           ← SEU dataset, classes, and transfer-task design
├── CITATION.cff              ← machine-readable citation metadata
├── .gitignore                ← ready for when code is added
└── LICENSE                   ← CC BY 4.0 for docs + IEEE copyright note for figures
```

---

## Roadmap

- [x] Public documentation of the method, challenges, and dataset
- [x] Key figures extracted from the paper
- [x] Machine-readable citation (`CITATION.cff`)
- [ ] Release training / inference source code
- [ ] Data preprocessing & task-split scripts
- [ ] Pretrained weights for the reported transfer tasks
- [ ] Reproduction guide (environment + commands)

> The code is **not yet public**. This repo will be updated as components are released.

---

## Citation

If you find this work useful, please cite the paper:

```bibtex
@inproceedings{kavianpour2022dcan,
  title     = {An Intelligent Gearbox Fault Diagnosis under Different Operating Conditions using Adversarial Domain Adaptation},
  author    = {Kavianpour, Mohammadreza and Ghorvei, Mohammadreza and Kavianpour, Parisa and Ramezani, Amin and Beheshti, Mohammad T. H.},
  booktitle = {2022 8th International Conference on Control, Instrumentation and Automation (ICCIA)},
  year      = {2022},
  publisher = {IEEE},
  doi       = {10.1109/ICCIA54998.2022.9737160}
}
```

---

## License & figures

- **Documentation** in this repository (all `.md` files and text) is released under **[CC BY 4.0](LICENSE)**.
- **Figures** in `assets/` are reproduced from the published IEEE paper and remain **© 2022 IEEE**. They are included here for scholarly, non-commercial showcase purposes under the authors' rights and academic fair-use conventions. They are **not** covered by the CC BY 4.0 license above. Any reuse of these figures must follow [IEEE's copyright and reuse policy](https://www.ieee.org/publications/rights/index.html). See [LICENSE](LICENSE) for the full notice.
