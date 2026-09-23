# Model Selection & Audit Timeline

## 1. Overview and Pre-Registration Rules
To guarantee scientific integrity and prevent adaptive overfitting:
- Architecture design, feature extraction, graph cutoffs, and hyperparameter tuning were conducted strictly on the **frozen 70:15:15 grouped development split** (`splits/tensor_development_split.json`).
- Outer folds were partitioned prior to any outer-fold model training.
- Hyperparameters and architectures were completely frozen before outer evaluation.

## 2. Chronological Decision Log

| Date & Time (UTC) | Phase | Partition Viewed | Decision Made | Adaptive Overfitting Risk |
| :--- | :--- | :--- | :--- | :--- |
| **2026-09-23 18:30** | Split Generation | Dataset-wide metadata | Generated 7 cryptographic split files (1 Protocol A, 1 Development, 5 Outer Folds). Verified zero ID, chemical system, formula, and hash leakage. | **None** (Unsupervised partition grouping) |
| **2026-09-23 19:15** | Protocol A | Train/Val IDs | Trained A0-A5 on 9,492 available training IDs with early stopping on validation. Evaluated test set once after configuration freeze. | **None** (Predefined official IDs) |
| **2026-09-23 19:45** | Protocol B Dev | Development Train/Val | Designed $E(3)$-equivariant tensor architecture with `e3nn` ($l=2e$ irreps) and Invariant GNN control. | **Contained to Dev Split** (Reported as development estimate) |
| **2026-09-23 20:25** | Protocol B Dev | Development Test | Evaluated B0-B5 on 14,225 test sites. Observed 29% Frobenius error reduction over invariant baseline. | **Low** (Frozen development split) |
| **2026-09-23 20:35** | Outer Fold 0 | Fold 0 Train/Val | Executed Fold 0 baseline training. Replaced BatchNorm1d with LayerNorm in Invariant GNN to handle batch size edge cases. | **Low** (Numerical stability bugfix applied universally) |
| **2026-09-24 02:30** | Adversarial Audit | All raw files & splits | Corrected unit attribution ($10^{21}\text{ V m}^{-2} = 10\text{ V \AA}^{-2}$), identified missing `JVASP-51` benchmark target, added scale-aware degeneracy masks, and removed B0 undefined orientation artifacts. | **None** (Forensic audit of existing files) |
