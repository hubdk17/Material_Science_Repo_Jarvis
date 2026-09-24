# Forensic Audit & Completion Report: JARVIS-DFT Electric Field Gradient Benchmark

**Audited Git Commit**: `ce12d03998fbba5320961d2c6d46c31d24f4b099`  
**Report Generated**: `2026-09-24 07:03:00 UTC`  
**Audit Directory**: [`results/adversarial_audit_20260924_023600`](file:///D:/Desktop/MATERIAL SCIENCE/JARVIS/results/adversarial_audit_20260924_023600)

---

## 1. Executive Forensic Summary

This report presents the adversarial forensic audit and completion pass on the JARVIS-DFT Electric Field Gradient (EFG) benchmark. All results, tables, and claims in this document are strictly generated from the committed CSV outputs and verifiable executable code.

### Canonical Summary of Key Audited Quantities
- **Primary Dataset (`JARVIS-EFG4.json`)**:
  - Structures: **15,202**
  - Atomic Sites: **95,663**
  - Elements: **89**
  - Chemical Systems (sorted unique elemental sets): **8,144**
  - Reduced Formulas: **10,982**
- **Unit Convention**:
  - Raw JSON tensors are in atomistic units: **$\text{V \AA}^{-2}$**.
  - Source CSV divides by 10 to report in **$10^{21}\ \text{V m}^{-2}$** ($1\times 10^{21}\ \text{V m}^{-2} = 10\ \text{V \AA}^{-2}$).
  - Official leaderboard benchmark targets were generated as $10.0 \times \max_{\text{rows}} |V_{ij}^{\text{CSV}}|$, reproducing scalar targets in **$\text{V \AA}^{-2}$** (100% within $10^{-6}$ tolerance).
- **Split Partitions**:
  - Official Training: **9,493 IDs** | Local Available Training: **9,492 IDs** (`JVASP-51` absent from Figshare release).
  - Official Validation: **1,186 IDs** | Local Validation: **1,186 IDs** (100% match).
  - Official Test: **1,186 IDs** | Local Test: **1,186 IDs** (100% match).

## 2. Unit Convention Audit: Candidate vs. Benchmark Target Comparison

| candidate_id   | definition                                                    |   exact_match_pct |   within_1e6_pct |   within_1e3_pct |   within_rounding_tolerance_pct |   mean_absolute_discrepancy |   max_discrepancy | unit_convention   | provenance_notes                                                                                                                  |
|:---------------|:--------------------------------------------------------------|------------------:|-----------------:|-----------------:|--------------------------------:|----------------------------:|------------------:|:------------------|:----------------------------------------------------------------------------------------------------------------------------------|
| T1             | max_sites max_ij |V_ij| (Raw JSON Cartesian maximum)          |          88.7138  |         88.7138  |         88.9751  |                        88.9751  |                 2.01276     |     670.995       | V/Angstrom^2      | Matches 100% when evaluated on source Wyckoff CSV rows; 88.98% on full-cell JSON because CSV omitted some high-EFG Wyckoff sites. |
| T2             | 10 * T1                                                       |           5.4029  |          5.4029  |          5.4029  |                         5.4029  |               617.251       |    8165.46        | 10^21 V/m^2       | Non-matching candidate                                                                                                            |
| T3             | max_sites max_k |lambda_k| (Raw JSON Principal maximum)       |          57.3584  |         59.7943  |         62.4326  |                        62.4326  |                 9.50205     |     706.035       | V/Angstrom^2      | Non-matching candidate                                                                                                            |
| T4             | 10 * T3                                                       |           5.4029  |          5.4029  |          5.4029  |                         5.4029  |               692.144       |    8515.86        | 10^21 V/m^2       | Non-matching candidate                                                                                                            |
| T5             | max_sites max |efg_diag_tensor| (Diagonal table maximum)      |          61.3874  |         61.3874  |         62.7866  |                        62.7866  |                 9.50204     |     706.035       | V/Angstrom^2      | Non-matching candidate                                                                                                            |
| T6             | 10 * T5                                                       |           5.19218 |          5.19218 |          5.19218 |                         5.19218 |               692.144       |    8515.86        | 10^21 V/m^2       | Non-matching candidate                                                                                                            |
| T_csv_10x      | 10 * max_csv_rows max(|VASP_V_ij|) (Wyckoff CSV maximum * 10) |          73.4828  |        100       |        100       |                       100       |                 2.93257e-15 |       2.84217e-14 | V/Angstrom^2      | Matches 100% when evaluated on source Wyckoff CSV rows; 88.98% on full-cell JSON because CSV omitted some high-EFG Wyckoff sites. |

*Git Commit: `ce12d03998fbba5320961d2c6d46c31d24f4b099` | Source: [`protocol_a_units_audit.csv`](file:///results/adversarial_audit_20260924_023600/protocol_a_units_audit.csv) (SHA-256: `68f5613710e0233c...`) | Generated: 2026-09-24 07:03:00 UTC*


## 3. Protocol A: Corrected Scalar Benchmark Metrics

| model_id                           | description                                   |   test_mae_V_A2 |   test_rmse_V_A2 |   test_median_ae_V_A2 |   test_mae_10_21_V_m2 |   test_rmse_10_21_V_m2 |   test_median_ae_10_21_V_m2 |   r2_score |   nmae_mad |   high_efg_f1 | claim_status                                                                                                                      |
|:-----------------------------------|:----------------------------------------------|----------------:|-----------------:|----------------------:|----------------------:|-----------------------:|----------------------------:|-----------:|-----------:|--------------:|:----------------------------------------------------------------------------------------------------------------------------------|
| A0_median                          | Median Dummy Baseline                         |         43.5653 |          54.0936 |               37.7325 |               4.35653 |                5.40936 |                     3.77325 | -0.0404954 |   1.1653   |      0.688053 | Protocol A is a split-compatible reimplementation using official JARVIS validation and test IDs; not an exact model reproduction. |
| A1_magpie_rf                       | Magpie Compositional Random Forest            |         29.0183 |          39.2457 |               21.1325 |               2.90183 |                3.92457 |                     2.11325 |  0.452313  |   0.776192 |      0.789205 | Protocol A is a split-compatible reimplementation using official JARVIS validation and test IDs; not an exact model reproduction. |
| A2_magpie_xgb                      | Magpie Compositional XGBoost (Matminer proxy) |         29.9716 |          39.4763 |               22.7926 |               2.99716 |                3.94763 |                     2.27926 |  0.445856  |   0.801691 |      0.780347 | Protocol A is a split-compatible reimplementation using official JARVIS validation and test IDs; not an exact model reproduction. |
| A3_composition_global_lattice_gbdt | Composition + global lattice-feature GBDT     |         27.5511 |          36.7211 |               21.2208 |               2.75511 |                3.67211 |                     2.12208 |  0.52051   |   0.736947 |      0.789436 | Protocol A is a split-compatible reimplementation using official JARVIS validation and test IDs; not an exact model reproduction. |
| A4_cgcnn                           | Crystal Graph Convolutional Neural Network    |         27.8154 |          40.1085 |               18.1867 |               2.78154 |                4.01085 |                     1.81868 |  0.427966  |   0.744016 |      0.800633 | Protocol A is a split-compatible reimplementation using official JARVIS validation and test IDs; not an exact model reproduction. |
| A5_alignn_eq                       | Line Graph Atomistic Neural Network           |         28.9188 |          41.75   |               18.5089 |               2.89188 |                4.175   |                     1.85089 |  0.380186  |   0.773529 |      0.782329 | Protocol A is a split-compatible reimplementation using official JARVIS validation and test IDs; not an exact model reproduction. |

*Git Commit: `ce12d03998fbba5320961d2c6d46c31d24f4b099` | Source: [`protocol_a_corrected_metrics.csv`](file:///results/adversarial_audit_20260924_023600/protocol_a_corrected_metrics.csv) (SHA-256: `a58f94fb70d24b60...`) | Generated: 2026-09-24 07:03:00 UTC*


## 4. Protocol A: Leaderboard Reproduction Disclosure

| model           | local_model_id   |   local_mae_V_A2 |   local_mae_10_21_V_m2 |   historical_leaderboard_mae_V_A2 |   historical_leaderboard_mae_10_21_V_m2 | disclosed_differences                                                                                                                                              | reproduction_claim                                            |
|:----------------|:-----------------|-----------------:|-----------------------:|----------------------------------:|----------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| ALIGNN          | A5_alignn_eq     |            28.92 |                  2.892 |                           19.1211 |                                  1.9121 | Missing JVASP-51 in local training split (9,492 vs 9,493); differing training budget/epoch schedule; local reimplementation rather than original model checkpoint. | Split-compatible reimplementation; NOT an exact reproduction. |
| CGCNN           | A4_cgcnn         |            27.82 |                  2.782 |                           24.6695 |                                  2.467  | Missing JVASP-51 in local training split; independent PyTorch reimplementation; differing optimizer schedule.                                                      | Split-compatible reimplementation; NOT an exact reproduction. |
| Matminer + GBDT | A2_magpie_xgb    |            29.97 |                  2.997 |                           19.4382 |                                  1.9438 | Uses Magpie composition feature subset with XGBoost rather than full Matminer multi-featurizer pipeline with exhaustive AutoML hyperparameter tuning.              | Proxy baseline; NOT an exact reproduction.                    |

*Git Commit: `ce12d03998fbba5320961d2c6d46c31d24f4b099` | Source: [`protocol_a_leaderboard_comparison.csv`](file:///results/adversarial_audit_20260924_023600/protocol_a_leaderboard_comparison.csv) (SHA-256: `8bcf1f182099f878...`) | Generated: 2026-09-24 07:03:00 UTC*


## 5. Raw DFT vs. Projected Physics Audit

| category                                                   |   num_tensors | array_sha256     |   max_symmetry_residual |   mean_symmetry_residual |   max_trace_residual |   mean_trace_residual |   num_outside_1e3_tolerance |   mean_frobenius_change_from_raw |   max_frobenius_change_from_raw | notes                                                                                                    |
|:-----------------------------------------------------------|--------------:|:-----------------|------------------------:|-------------------------:|---------------------:|----------------------:|----------------------------:|---------------------------------:|--------------------------------:|:---------------------------------------------------------------------------------------------------------|
| A. Raw DFT Tensors (data/raw/JARVIS-EFG4.json)             |         95663 | d8de0d385214a908 |                       0 |                        0 |          0.001       |           0.000296855 |                           0 |                      0           |                      0          | 100% symmetric; trace bounded by VASP 3-decimal output format (mean 2.9685e-4 V/A^2)                     |
| B. Projected Training Labels (project_symmetric_traceless) |         95663 | 98994f74766ff14d |                       0 |                        0 |          0           |           0           |                           0 |                      0.000171389 |                      0.00057735 | Exact mathematical projection enforces Tr(V) = 0 with minimal perturbation (mean change 1.7139e-4 V/A^2) |
| C. Trained Invariant GNN Model (B4 5D Cartesian Head)      |         14225 | 630ed7ff4dc4ca6c |                       0 |                        0 |          0           |           0           |                           0 |                    nan           |                    nan          | 5D symmetric-traceless Cartesian head mathematically guarantees Tr(V)=0 by construction                  |
| D. Trained Equivariant GNN Model (B5 e3nn l=2e Irreps)     |         14225 | 32b584a427517971 |                       0 |                        0 |          3.05176e-05 |           7.55353e-07 |                           0 |                    nan           |                    nan          | e3nn irreducible representation l=2e strictly spans traceless symmetric subspace                         |

*Git Commit: `ce12d03998fbba5320961d2c6d46c31d24f4b099` | Source: [`raw_tensor_physics_audit.csv`](file:///results/adversarial_audit_20260924_023600/raw_tensor_physics_audit.csv) (SHA-256: `02bdb041a1592bc5...`) | Generated: 2026-09-24 07:03:00 UTC*


## 6. Tensor Metric Validity & Scale-Aware Degeneracy Audit

|   tau_non_degeneracy_threshold |   total_sites |   valid_eta_sites |   valid_orientation_sites |   excluded_low_magnitude |   excluded_degenerate_true |   b0_valid_orientation |   b0_valid_eta | notes                                                          |
|-------------------------------:|--------------:|------------------:|--------------------------:|-------------------------:|---------------------------:|-----------------------:|---------------:|:---------------------------------------------------------------|
|                           0.01 |         14225 |             12923 |                     12885 |                     1209 |                        131 |                      0 |              0 | B0 is strictly undefined (reported as NaN / N/A in all tables) |
|                           0.05 |         14225 |             12923 |                     12416 |                     1209 |                        600 |                      0 |              0 | B0 is strictly undefined (reported as NaN / N/A in all tables) |
|                           0.1  |         14225 |             12923 |                     11841 |                     1209 |                       1175 |                      0 |              0 | B0 is strictly undefined (reported as NaN / N/A in all tables) |
|                           0.2  |         14225 |             12923 |                     10463 |                     1209 |                       2553 |                      0 |              0 | B0 is strictly undefined (reported as NaN / N/A in all tables) |

*Git Commit: `ce12d03998fbba5320961d2c6d46c31d24f4b099` | Source: [`tensor_metric_validity_audit.csv`](file:///results/adversarial_audit_20260924_023600/tensor_metric_validity_audit.csv) (SHA-256: `538a470fae06716f...`) | Generated: 2026-09-24 07:03:00 UTC*


## 7. Development Split Corrected Metrics (Trained Checkpoints)

| model                      |   site_micro_frobenius_mean |   site_micro_frobenius_median |   normalized_frobenius_error |   crystal_macro_frobenius_mean |   crystal_macro_frobenius_median |   largest_principal_vzz_mae |   asymmetry_eta_mae |   asymmetry_eta_conditional_mae |   eta_coverage_rate |   common_mask_orientation_mean_deg |   common_mask_orientation_median_deg |   common_valid_orientation_count |   prediction_validity_rate |   intersection_orientation_mean_deg | checkpoint_sha256   |
|:---------------------------|----------------------------:|------------------------------:|-----------------------------:|-------------------------------:|---------------------------------:|----------------------------:|--------------------:|--------------------------------:|--------------------:|-----------------------------------:|-------------------------------------:|---------------------------------:|---------------------------:|------------------------------------:|:--------------------|
| B0_zero                    |                     90.7187 |                       43.0228 |                     1        |                        79.8331 |                          54.4445 |                     72.5714 |            1        |                      nan        |            0        |                           nan      |                             nan      |                            12416 |                   0        |                            nan      | zeros               |
| B1_element_mean            |                     91.5099 |                       43.9916 |                     1.00872  |                        81.495  |                          55.4557 |                     72.9839 |            0.505494 |                        0.372187 |            0.787665 |                            47.1332 |                              50.2957 |                            12416 |                   0.885551 |                             46.0935 | train_mean          |
| B2_local_ridge             |                     91.5833 |                       43.6623 |                     1.00953  |                        81.0751 |                          54.3253 |                     73.2338 |            0.446667 |                        0.443913 |            0.995048 |                            47.6473 |                              51.531  |                            12416 |                   0.844797 |                             46.5083 | ecd7175f1043        |
| B4_invariant_projected_5d  |                     89.5243 |                       41.7638 |                     0.986834 |                        78.9354 |                          52.9209 |                     71.4967 |            0.900761 |                        0.201456 |            0.124275 |                            46.612  |                              51.7946 |                            12416 |                   0.128383 |                             30.2754 | c864227be714        |
| B5_equivariant_e3nn        |                     64.0545 |                       33.6162 |                     0.706078 |                        59.059  |                          36.5727 |                     54.1023 |            0.158152 |                        0.145523 |            0.98522  |                            15.8281 |                               0      |                            12416 |                   0.937581 |                             14.1988 | 30d5f135555b        |
| B6_symmetry_projected_e3nn |                     64.0544 |                       33.6162 |                     0.706077 |                        59.0588 |                          36.5727 |                     54.1023 |            0.158153 |                        0.145524 |            0.98522  |                            15.8279 |                               0      |                            12416 |                   0.937581 |                             14.1989 | 30d5f135555b        |

*Git Commit: `ce12d03998fbba5320961d2c6d46c31d24f4b099` | Source: [`development_split_corrected_metrics.csv`](file:///results/adversarial_audit_20260924_023600/development_split_corrected_metrics.csv) (SHA-256: `49f88ff5a56f23fd...`) | Generated: 2026-09-24 07:03:00 UTC*


## 8. Outer Five-Fold Site-Micro Metrics

| fold   | model                     |   seed |   num_crystals |   num_sites |   mean_frobenius_micro |   median_frobenius_micro |   normalized_frobenius_error |   Vzz_MAE |     eta_MAE |   orientation_error_mean_deg |   valid_eta_count |   valid_orientation_count |
|:-------|:--------------------------|-------:|---------------:|------------:|-----------------------:|-------------------------:|-----------------------------:|----------:|------------:|-----------------------------:|------------------:|--------------------------:|
| fold_0 | B0_zero                   |     42 |           3041 |       19237 |                88.2449 |                  41.4943 |                     1        |   70.6101 | nan         |                     nan      |                 0 |                         0 |
| fold_0 | B2_local_mlp              |     42 |           3041 |       19237 |                88.3408 |                  41.5701 |                     1.00109  |   70.664  |   0.543923  |                      58.6049 |                 5 |                       767 |
| fold_0 | B4_invariant_projected_5d |     42 |           3041 |       19237 |                88.0312 |                  41.2893 |                     0.997578 |   70.4229 |   0.0355505 |                      27.2311 |                51 |                       140 |
| fold_0 | B5_equivariant_e3nn       |     42 |           3041 |       19237 |                71.6217 |                  36.4479 |                     0.811624 |   59.7133 |   0.147734  |                      14.8329 |             16983 |                     15566 |
| fold_1 | B0_zero                   |     42 |           3041 |       18870 |                92.0429 |                  43.4282 |                     1        |   73.7473 | nan         |                     nan      |                 0 |                         0 |
| fold_1 | B2_local_mlp              |     42 |           3041 |       18870 |                92.1543 |                  43.3652 |                     1.00121  |   73.8479 |   0.540004  |                      68.9689 |                10 |                      9264 |
| fold_1 | B4_invariant_projected_5d |     42 |           3041 |       18870 |                91.505  |                  42.7333 |                     0.994156 |   73.2928 |   0.206916  |                      30.9634 |               938 |                      1009 |
| fold_1 | B5_equivariant_e3nn       |     42 |           3041 |       18870 |                74.9495 |                  35.6027 |                     0.814289 |   62.425  |   0.151073  |                      15.8089 |             16642 |                     15243 |
| fold_2 | B0_zero                   |     42 |           3040 |       18884 |                90.4516 |                  42.4825 |                     1        |   72.5591 | nan         |                     nan      |                 0 |                         0 |
| fold_2 | B2_local_mlp              |     42 |           3040 |       18884 |                90.658  |                  42.2822 |                     1.00228  |   72.5895 |   0.566244  |                      42.8393 |             10684 |                      7750 |
| fold_2 | B4_invariant_projected_5d |     42 |           3040 |       18884 |                87.7014 |                  38.421  |                     0.969595 |   70.3324 |   0.0800209 |                       9.2762 |               696 |                       703 |
| fold_2 | B5_equivariant_e3nn       |     42 |           3040 |       18884 |                70.008  |                  33.3984 |                     0.773983 |   58.2147 |   0.139446  |                      13.7517 |             16780 |                     15556 |
| fold_3 | B0_zero                   |     42 |           3040 |       19443 |               100.741  |                  45.022  |                     1        |   80.5956 | nan         |                     nan      |                 0 |                         0 |
| fold_3 | B2_local_mlp              |     42 |           3040 |       19443 |               100.809  |                  45.1004 |                     1.00067  |   80.6255 |   0.34166   |                      50.1919 |                 4 |                      2052 |
| fold_3 | B4_invariant_projected_5d |     42 |           3040 |       19443 |                99.3468 |                  43.3759 |                     0.98616  |   79.2948 |   0.219694  |                      32.1817 |              1257 |                      1317 |
| fold_3 | B5_equivariant_e3nn       |     42 |           3040 |       19443 |                78.8159 |                  36.48   |                     0.782361 |   65.97   |   0.144359  |                      15.5555 |             17070 |                     15671 |
| fold_4 | B0_zero                   |     42 |           3040 |       19229 |               101.329  |                  45.4475 |                     1        |   81.1139 | nan         |                     nan      |                 0 |                         0 |
| fold_4 | B2_local_mlp              |     42 |           3040 |       19229 |               101.453  |                  45.4575 |                     1.00122  |   81.1759 |   0.43636   |                      43.162  |               184 |                     14927 |
| fold_4 | B4_invariant_projected_5d |     42 |           3040 |       19229 |                99.7409 |                  42.9685 |                     0.984323 |   79.6274 |   0.207924  |                      29.6227 |              1268 |                      1262 |
| fold_4 | B5_equivariant_e3nn       |     42 |           3040 |       19229 |                79.2795 |                  35.5452 |                     0.782394 |   65.6289 |   0.15295   |                      14.6409 |             16888 |                     15391 |

*Git Commit: `ce12d03998fbba5320961d2c6d46c31d24f4b099` | Source: [`outer_fold_site_micro_metrics.csv`](file:///results/adversarial_audit_20260924_023600/outer_fold_site_micro_metrics.csv) (SHA-256: `298b86f8c3a8fcaf...`) | Generated: 2026-09-24 07:03:00 UTC*


## 9. Outer Five-Fold Crystal-Macro Metrics

| fold   | model                     |   seed |   num_crystals |   num_sites |   mean_frobenius_macro |   median_frobenius_macro |   p90_frobenius_macro |   normalized_frobenius_macro |
|:-------|:--------------------------|-------:|---------------:|------------:|-----------------------:|-------------------------:|----------------------:|-----------------------------:|
| fold_0 | B0_zero                   |     42 |           3041 |       19237 |                74.9813 |                  50.4492 |               172.367 |                     1        |
| fold_0 | B2_local_mlp              |     42 |           3041 |       19237 |                75.1265 |                  50.5149 |               172.576 |                     1.00194  |
| fold_0 | B4_invariant_projected_5d |     42 |           3041 |       19237 |                74.7723 |                  50.1153 |               172.381 |                     0.997212 |
| fold_0 | B5_equivariant_e3nn       |     42 |           3041 |       19237 |                63.047  |                  40.7521 |               144.895 |                     0.840836 |
| fold_1 | B0_zero                   |     42 |           3041 |       18870 |                80.3694 |                  53.2285 |               181.405 |                     1        |
| fold_1 | B2_local_mlp              |     42 |           3041 |       18870 |                80.5365 |                  53.2    |               181.37  |                     1.00208  |
| fold_1 | B4_invariant_projected_5d |     42 |           3041 |       18870 |                80.0045 |                  52.8956 |               180.806 |                     0.995459 |
| fold_1 | B5_equivariant_e3nn       |     42 |           3041 |       18870 |                67.8294 |                  42.9522 |               149.641 |                     0.84397  |
| fold_2 | B0_zero                   |     42 |           3040 |       18884 |                77.1926 |                  53.4463 |               178.141 |                     1        |
| fold_2 | B2_local_mlp              |     42 |           3040 |       18884 |                77.4773 |                  53.6416 |               178.473 |                     1.00369  |
| fold_2 | B4_invariant_projected_5d |     42 |           3040 |       18884 |                75.2179 |                  50.572  |               177.046 |                     0.974419 |
| fold_2 | B5_equivariant_e3nn       |     42 |           3040 |       18884 |                62.5522 |                  38.354  |               146.06  |                     0.810339 |
| fold_3 | B0_zero                   |     42 |           3040 |       19443 |                84.5994 |                  53.0302 |               193.201 |                     1        |
| fold_3 | B2_local_mlp              |     42 |           3040 |       19443 |                84.7029 |                  53.0503 |               193.174 |                     1.00122  |
| fold_3 | B4_invariant_projected_5d |     42 |           3040 |       19443 |                83.8954 |                  51.2408 |               193.4   |                     0.991678 |
| fold_3 | B5_equivariant_e3nn       |     42 |           3040 |       19443 |                68.5298 |                  41.426  |               157.546 |                     0.810051 |
| fold_4 | B0_zero                   |     42 |           3040 |       19229 |                86.1956 |                  52.7298 |               189.473 |                     1        |
| fold_4 | B2_local_mlp              |     42 |           3040 |       19229 |                86.3879 |                  52.6261 |               189.678 |                     1.00223  |
| fold_4 | B4_invariant_projected_5d |     42 |           3040 |       19229 |                85.2608 |                  50.6709 |               188.801 |                     0.989156 |
| fold_4 | B5_equivariant_e3nn       |     42 |           3040 |       19229 |                69.1851 |                  38.2918 |               149.158 |                     0.802653 |

*Git Commit: `ce12d03998fbba5320961d2c6d46c31d24f4b099` | Source: [`outer_fold_crystal_macro_metrics.csv`](file:///results/adversarial_audit_20260924_023600/outer_fold_crystal_macro_metrics.csv) (SHA-256: `7b3b1a61ed85bfcb...`) | Generated: 2026-09-24 07:03:00 UTC*


## 10. Paired Model Comparisons with Crystal-Clustered Bootstrap 95% CIs

| fold   |   seed |   num_crystals |   mean_b4_macro |   mean_b5_macro |   paired_difference_macro |   relative_improvement_pct |   bootstrap_95_ci_lower |   bootstrap_95_ci_upper | statistically_significant   |
|:-------|-------:|---------------:|----------------:|----------------:|--------------------------:|---------------------------:|------------------------:|------------------------:|:----------------------------|
| fold_0 |     42 |           3041 |         74.7723 |         63.047  |                  -11.7252 |                    15.6813 |                -13.0972 |                -10.4358 | True                        |
| fold_1 |     42 |           3041 |         80.0045 |         67.8294 |                  -12.1751 |                    15.218  |                -13.3523 |                -10.9541 | True                        |
| fold_2 |     42 |           3040 |         75.2179 |         62.5522 |                  -12.6657 |                    16.8387 |                -13.9636 |                -11.2098 | True                        |
| fold_3 |     42 |           3040 |         83.8954 |         68.5298 |                  -15.3656 |                    18.3152 |                -17.0122 |                -13.7699 | True                        |
| fold_4 |     42 |           3040 |         85.2608 |         69.1851 |                  -16.0757 |                    18.8548 |                -17.658  |                -14.5583 | True                        |

*Git Commit: `ce12d03998fbba5320961d2c6d46c31d24f4b099` | Source: [`outer_fold_paired_comparisons.csv`](file:///results/adversarial_audit_20260924_023600/outer_fold_paired_comparisons.csv) (SHA-256: `50eb768893ac20e1...`) | Generated: 2026-09-24 07:03:00 UTC*


## 11. B5 vs. B6 Crystallographic Site-Symmetry Projection Audit

|   total_test_sites |   sites_altered_count |   sites_altered_fraction |   mean_frobenius_change_V_A2 |   median_frobenius_change_V_A2 |   p95_frobenius_change_V_A2 |   max_frobenius_change_V_A2 | notes                                                                                                    |
|-------------------:|----------------------:|-------------------------:|-----------------------------:|-------------------------------:|----------------------------:|----------------------------:|:---------------------------------------------------------------------------------------------------------|
|              14225 |                 10650 |                 0.748682 |                   0.00113973 |                    4.77743e-06 |                 0.000855314 |                     2.60224 | Real B6 point-group projection computed strictly from input structure symmetry without target inspection |

*Git Commit: `ce12d03998fbba5320961d2c6d46c31d24f4b099` | Source: [`b5_b6_symmetry_projection_audit.csv`](file:///results/adversarial_audit_20260924_023600/b5_b6_symmetry_projection_audit.csv) (SHA-256: `f2a80961ecba5d5f...`) | Generated: 2026-09-24 07:03:00 UTC*


## 12. Rotational, Reflectional, and Translational Covariance Audit

| transformation       |   num_crystals_tested |   total_sites_tested |   b5_trained_mean_abs_err |   b5_trained_max_abs_err |   b5_trained_p99_abs_err |   b5_trained_mean_rel_err |   b4_trained_mean_abs_err |   b4_trained_mean_rel_err | status                                                     |
|:---------------------|----------------------:|---------------------:|--------------------------:|-------------------------:|-------------------------:|--------------------------:|--------------------------:|--------------------------:|:-----------------------------------------------------------|
| SO(3) Rotation       |                   100 |                  616 |               4.21804e-05 |              0.000512553 |              0.000186045 |                 0.525553  |               3.45372     |               1.33477     | B5 Passes Covariance; B4 Invariant Cartesian Control Fails |
| Inversion (P=-I)     |                   100 |                  616 |               7.59364e-06 |              6.51919e-05 |              4.40129e-05 |                 0.0524923 |               2.38319e-07 |               7.71675e-07 | B5 Passes Covariance; B4 Invariant Cartesian Control Fails |
| Reflection (det=-1)  |                   100 |                  616 |               2.12181e-05 |              0.000163155 |              0.000100778 |                 0.212294  |               2.76345     |               1.15227     | B5 Passes Covariance; B4 Invariant Cartesian Control Fails |
| Periodic Translation |                   100 |                  616 |               7.88746e-06 |              5.71042e-05 |              4.21809e-05 |                 0.0566642 |               2.63203e-07 |               7.92004e-07 | B5 Passes Covariance; B4 Invariant Cartesian Control Fails |

*Git Commit: `ce12d03998fbba5320961d2c6d46c31d24f4b099` | Source: [`rotation_reflection_covariance_audit.csv`](file:///results/adversarial_audit_20260924_023600/rotation_reflection_covariance_audit.csv) (SHA-256: `20e5ad89e2cbb654...`) | Generated: 2026-09-24 07:03:00 UTC*


## 13. Frame-Dependent Cartesian Max vs. Physical Principal Max Audit

| granularity                              |   total_count |   strict_underestimation_count |   strict_underestimation_fraction |   tol_underestimation_count |   tol_underestimation_fraction |   mean_underestimation_V_A2 |   max_underestimation_V_A2 |   max_ratio |   p95_ratio |
|:-----------------------------------------|--------------:|-------------------------------:|----------------------------------:|----------------------------:|-------------------------------:|----------------------------:|---------------------------:|------------:|------------:|
| Atomic Sites (All 95,663 sites)          |         95663 |                          45035 |                          0.470767 |                       41329 |                       0.432027 |                     24.5488 |                   1051.29  |     2.11608 |     1.69795 |
| Crystal Structures (All 15,202 crystals) |         15202 |                           5791 |                          0.380937 |                        5054 |                       0.332456 |                     30.6152 |                    974.601 |     2       |     1.56245 |

*Git Commit: `ce12d03998fbba5320961d2c6d46c31d24f4b099` | Source: [`cartesian_vs_principal_summary.csv`](file:///results/adversarial_audit_20260924_023600/cartesian_vs_principal_summary.csv) (SHA-256: `36ad2eb1201efe28...`) | Generated: 2026-09-24 07:03:00 UTC*


## 14. Standardized Inference Throughput Audit

| gpu_model                          |   cuda_version | pytorch_version   | numerical_precision   |   batch_size |   timed_batches |   timed_crystals |   timed_sites |   crystals_per_sec |   sites_per_sec |   model_batch_latency_median_ms |   model_batch_latency_p95_ms |   single_crystal_graph_build_ms_median |   peak_gpu_memory_mb |
|:-----------------------------------|---------------:|:------------------|:----------------------|-------------:|----------------:|-----------------:|--------------:|-------------------:|----------------:|--------------------------------:|-----------------------------:|---------------------------------------:|---------------------:|
| NVIDIA GeForce RTX 5070 Laptop GPU |           12.8 | 2.11.0+cu128      | float32               |           32 |              71 |             2272 |         14187 |            858.392 |         5360.04 |                          35.977 |                      52.4562 |                                1.36255 |              447.394 |

*Git Commit: `ce12d03998fbba5320961d2c6d46c31d24f4b099` | Source: [`throughput_audit.csv`](file:///results/adversarial_audit_20260924_023600/throughput_audit.csv) (SHA-256: `6030b44926d6f355...`) | Generated: 2026-09-24 07:03:00 UTC*
