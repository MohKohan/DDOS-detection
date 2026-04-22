# ML & Deep Learning for DDoS Detection — MGR820

> **Adapted and extended from:** [mvoassis/ml-dl-ddos-detection](https://github.com/mvoassis/ml-dl-ddos-detection) by Marcos V. O. Assis  
> **Original paper:** *A GRU deep learning system against attacks in software defined networks* — [DOI: 10.1016/j.jnca.2020.102942](https://doi.org/10.1016/j.jnca.2020.102942)

---

## Overview

This notebook was developed as part of the **MGR820** course project. It builds on the publicly available baseline code by Marcos V. O. Assis, which evaluates machine learning and deep learning methods for binary DDoS detection using the CIC-DDoS2019 dataset.

The original framework and dataset pipeline are credited to the original author. My contributions focus on **model architecture modernization**, **additional evaluation metrics**, and **improved visualization**.

---

## Dataset

**CIC-DDoS2019** — Canadian Institute for Cybersecurity  
- Source: https://www.unb.ca/cic/datasets/ddos-2019.html  
- **Training (Day 1 — Jan 12):** DrDoS_DNS, DrDoS_LDAP, DrDoS_MSSQL, DrDoS_NetBIOS, DrDoS_NTP, DrDoS_SNMP, DrDoS_SSDP, DrDoS_UDP, Syn, TFTP, UDPLag  
- **Testing (Day 2 — Mar 11):** LDAP, MSSQL, NetBIOS, Portmap, Syn  
- **Task:** Binary classification — BENIGN (0) vs. DDoS Attack (1)

---

## Methods Evaluated

| Category | Methods |
|---|---|
| Deep Learning | GRU, LSTM, CNN, DNN |
| Classical ML | SVM, Logistic Regression, Gradient Descent (SGD), k-Nearest Neighbors |

---

## My Contributions vs. the Original

The baseline notebook provides the data loading pipeline, auxiliary functions, and the original model stubs. The following changes were made in this version:

### 1. Modernized Deep Learning Architectures (TensorFlow/Keras)

The original model definitions used the legacy `keras` API with minimal architectures. This version rewrites all four DL models using `tensorflow.keras` with the explicit `Input()` layer API and improved capacity:

| Model | Original | This version |
|---|---|---|
| **GRU** | GRU(32) + Dropout(0.5) + Dense(10) | GRU(64) + Dropout(0.3) + Dense(64) |
| **LSTM** | LSTM(32) + Dropout(0.5) + Dense(10) | LSTM(64) + Dropout(0.3) + Dense(64) |
| **CNN** | 3× Conv1D + MaxPool + Dense(10) | Conv1D(64) + **BatchNormalization** + Dropout + Dense(64) |
| **DNN** | Dense(2) + Dense(1) *(minimal)* | Dense(128) → Dense(64) → Dense(32) → Dense(1) *(proper deep network)* |

### 2. False Positive Rate as an Evaluation Metric

The original code only reports Accuracy, Precision, Recall, F1, and detection rates. This version adds **False Positive Rate (FPR)** for every model:

```python
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
```

FPR is a critical security metric — it measures how often legitimate traffic is wrongly flagged as an attack, which directly impacts operational usability.

### 3. Confusion Matrix Heatmaps for Every Model

The original outputs no confusion matrix visualizations. This version adds seaborn heatmap plots for all 8 models (GRU, CNN, LSTM, DNN, SVM, LR, GD, kNN), making it easier to visually inspect class-level error distributions.

### 4. Improved Logistic Regression Prediction

The original uses `model_lr.predict()` (hard class labels). This version uses probability thresholding:

```python
y_pred_lr = model_lr.predict_proba(X_test)[:, 1]
y_pred_lr = (y_pred_lr > 0.5).astype(int)
```

This is more principled and allows the threshold to be adjusted for different sensitivity/specificity trade-offs.

### 5. Defensive `test_normal_atk()` for CNN

A more robust local version of `test_normal_atk()` was implemented in the CNN block that safely handles cases where a class may be absent from the wrong-prediction set, preventing index errors.

### 6. Code Documentation

Inline explanations were added to the `load_file()` and `load_huge_file()` functions to clarify the random row sampling strategy and the chunk-based processing logic.

---

## What Was NOT Changed

To be transparent: the following are taken directly from the original codebase with no modification:

- `load_file()` and `load_huge_file()` function logic
- All data preprocessing (timestamp hashing, label encoding, infinity replacement)
- `train_test()`, `normalize_data()`, `format_3d()`, `format_2d()` utility functions
- `compile_train()` and `testes()` training/evaluation functions
- `save_model()` / `load_model()` persistence functions
- Dataset upsampling pipeline
- All results visualization (bar plots, catplots)
- Textual analysis and interpretation

---

## Environment

```
Python      3.x (Kaggle kernel)
TensorFlow  2.x (tensorflow.keras)
scikit-learn
Pandas, NumPy, Matplotlib, Seaborn
```

---

## How to Run

1. Upload the CIC-DDoS2019 dataset to Kaggle (or modify paths for local use)  
2. Run the data loading and preprocessing cells in order  
3. Run the model training section  
4. Run each model's testing cell to generate metrics and confusion matrices  
5. View the `results` DataFrame for a summary comparison

---

## Acknowledgements

This work is based on the open-source notebook by **Marcos V. O. Assis** (mvoassis@gmail.com), published alongside the paper:

> Assis, M.V.O. et al. *A GRU deep learning system against attacks in software defined networks.* Journal of Network and Computer Applications, 2021. https://doi.org/10.1016/j.jnca.2020.102942

---

## License

This adapted version is shared for academic and educational purposes. The original code and methodology belong to the original author. Please cite the original paper if you use this work in research.