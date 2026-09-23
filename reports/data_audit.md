# Comprehensive Data Audit Report: JARVIS-DFT EFG Dataset

## 1. Overview
- **Total Structures**: 15,202
- **Total Sites**: 95,663
- **Unique JARVIS IDs**: 15,202
- **Unique Chemical Systems**: 8,144
- **Unique Reduced Formulas**: 10,982

## 2. Integrity and Missingness
| Integrity Check | Count | Status |
| :--- | :--- | :--- |
| Missing / Null JSON records | 0 | **PASS** |
| NaN or Infinite values | 0 | **PASS** |
| Singular / Invertible Lattices | 0 | **PASS** |
| Site count mismatches (elements vs coords vs tensors) | 0 | **PASS** |
| Duplicate JARVIS IDs | 0 | **PASS** |

## 3. Physical Tensor Validation
- **Symmetry ($||V - V^T||_F$)**: Exactly zero across all sites (max: 0.000000).
- **Trace Residual ($|\mathrm{Tr}(V)|$)**: Maximum 1.0000e-03, Mean 2.9685e-04 (strictly bounded by VASP 3-decimal rounding).
- **Zero-EFG Sites**: 5,227 sites (5.46%) correspond to high-symmetry local atomic environments (e.g. cubic point groups).
- **Frobenius Norm**: Min = 0.00, Median = 43.40, 99th percentile = 690.21, Max = 3001.80 V/Å².

## 4. Duplicate Structures & Polymorphs
- **Identified Duplicate Structure Groups**: 209 groups (423 structures total).
- In Protocol B, all structures within the same chemical system group are assigned to the exact same outer fold, preventing any train-test leakage.

## 5. Summary Table
| Metric | Value |
| :--- | :--- |
| Total Structures | 15202 |
| Unique JARVIS IDs | 15202 |
| Total Atomic Sites | 95663 |
| Unique Chemical Systems | 8144 |
| Unique Reduced Formulas | 10982 |
| Duplicate Structure Groups | 209 |
| Total Structures in Duplicate Groups | 423 |
| Zero Tensor Sites (Frobenius < 1e-6) | 5227 |
| Zero Tensor Sites (%) | 5.46% |
| Max Symmetry Residual (||V - V^T||_F) | 0.000000e+00 |
| Mean Symmetry Residual | 0.000000e+00 |
| Max Trace Residual (|Tr(V)|) | 1.000000e-03 |
| Mean Trace Residual | 2.968546e-04 |
| Max Eigenvalue Diff (VASP vs numpy) | 1.447269e-03 |
| Frobenius Norm Median | 43.4025 |
| Frobenius Norm 99th Percentile | 690.2150 |
| Frobenius Norm Max | 3001.8021 |
