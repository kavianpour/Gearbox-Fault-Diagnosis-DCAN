# Dataset — SEU Gearbox Dataset & transfer-task design

This document describes the dataset used in the paper, the fault classes, the cross-condition transfer tasks, and how to obtain the data.

---

## 1. The SEU gearbox dataset

The experiments use the gearbox dataset developed by **Southeast University (SEU), China** — one of the most widely used datasets for cross-condition rotating-machinery diagnosis. Relatively few gearbox datasets cover *multiple* operating conditions, which is why this one is well suited to domain-adaptation studies.

| Property | Detail |
|---|---|
| Source | Southeast University (SEU), China |
| Acquisition rig | **Drivetrain Dynamics Simulator (DDS)** |
| Channels | 8 channels of vibration information |
| Sub-datasets | Two: a **bearing** sub-dataset and a **gearbox** sub-dataset |
| Channel used in this paper | **Channel 2** of the gearbox sub-dataset |
| Preprocessing | **z-score** normalization of the raw 1-D signal |
| Train/test split | **80 % / 20 %** |
| Label availability | Source domain **labeled**; target domain **unlabeled** |

---

## 2. Health classes (Table II in the paper)

The gearbox sub-dataset contains **one healthy** condition and **four faulty** conditions — five classes total:

| Label | Type | Description |
|---|---|---|
| **C0** | Health | Normal condition |
| **C1** | Chipped | Crack appears in the gear feet |
| **C2** | Root | Crack appears in the root of the gear feet |
| **C3** | Miss | Missing feet in the gear |
| **C4** | Surface | Wear appears on the surface of the gear |

---

## 3. Operating conditions & transfer tasks

The dataset provides **two operating conditions** defined by rotational speed and load:

| Task | Rotational speed | Load |
|---|---|---|
| **A** | 20 Hz | 0 V |
| **B** | 30 Hz | 2 V |

Because there are two conditions, there are **two transfer directions**, each treating one condition as the labeled source and the other as the unlabeled target:

- **A → B** — train on labeled task A, adapt to and test on unlabeled task B.
- **B → A** — train on labeled task B, adapt to and test on unlabeled task A.

In every transfer task the source is labeled and the target is unlabeled, mirroring the realistic scenario where the deployment condition cannot be annotated. Reported accuracies for both directions are in the [Headline results](../README.md#headline-results).

---

## 4. Obtaining the data

The SEU gearbox dataset is publicly distributed by its original authors. It is **not redistributed in this repository.** The canonical release is the Southeast University "Mechanical-datasets" / Gearbox Dataset, commonly mirrored on GitHub and data-sharing platforms.

To use it with this work:

1. Obtain the SEU **gearbox** sub-dataset from the original distribution.
2. Select **channel 2**.
3. Apply **z-score** normalization to the raw vibration signal.
4. Build the two tasks (A: 20 Hz–0 V, B: 30 Hz–2 V) and split each **80 / 20** for train / test.
5. For a transfer task, keep source labels and treat the target as unlabeled.

> Always check and comply with the licensing/usage terms attached to the dataset by its original authors.

---

*This document is part of the documentation-only release; preprocessing and task-split scripts will accompany the code when it is released. See the [Roadmap](../README.md#roadmap).*
