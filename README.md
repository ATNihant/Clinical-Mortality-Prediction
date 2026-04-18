# 🧠 MIMIC-III Clinical Mortality Prediction: GNN + Symbolic AI Hybrid

A **neuro-symbolic hybrid pipeline** for predicting in-hospital ICU mortality by fusing Graph Neural Networks with interpretable Prolog-based clinical rules.

---

## 📌 Overview

This project combines two complementary AI paradigms to tackle the class-imbalanced, high-stakes problem of ICU mortality prediction:

| Component | Role |
|---|---|
| **Graph Convolutional Network (GCN)** | Learns latent patient similarity embeddings from a population graph |
| **Symbolic Prolog Rules** | Encodes interpretable clinical domain knowledge (renal, sepsis, metabolic risk) |
| **Hybrid Fusion Classifier** | Combines both signal types for final mortality prediction |

The ablation study included in the notebook validates that the **hybrid outperforms either component alone**.

---

## 🏗️ Architecture

```
MIMIC-III Tables
     │
     ▼
Feature Engineering  ──►  Patient Population Graph
(labs + ICD-9 codes)            (cosine similarity edges)
                                     │
                                     ▼
                             2-Layer GCN
                          (GCNConv → GCNConv)
                                     │
                                32-dim Embeddings
                                     │
             ┌───────────────────────┘
             │
  Prolog Rule Engine  ──►  5 Clinical Rule Scores
  (renal / sepsis /              │
   metabolic risk)               ▼
                         Hybrid Feature Vector (37-dim)
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
          Logistic Regression         Gradient Boosting
            (interpretable)            (high performance)
```

---

## 📂 Dataset

**[MIMIC-III Clinical Database Demo v1.4](https://physionet.org/content/mimiciii-demo/1.4/)** — available on Kaggle.

> ⚠️ Full MIMIC-III access requires credentialed PhysioNet authorization. This notebook uses the publicly available demo subset.

The following tables are used:

| Table | Key Columns |
|---|---|
| `ADMISSIONS` | `hadm_id`, `hospital_expire_flag`, `ethnicity`, `insurance` |
| `PATIENTS` | `subject_id`, `gender`, `dob` |
| `DIAGNOSES_ICD` | `icd9_code` (top-20 codes as binary features) |
| `LABEVENTS` | creatinine (50912), WBC (51301), glucose (50931) |

---

## ⚙️ Pipeline Steps

### 1. Feature Engineering
- Lab pivot (mean creatinine / WBC / glucose per admission)
- Top-20 ICD-9 diagnosis codes as one-hot binary flags
- Mortality label from `hospital_expire_flag`
- Median imputation for missing lab values

### 2. Patient Population Graph
- **Nodes**: individual patient admissions with feature vectors
- **Edges**: cosine similarity > `0.75` threshold (scale-invariant, robust to mixed feature types)

### 3. GCN Training
- 2-layer GCNConv: `Input → 64 → 32 → logits`
- Dropout (0.3) and L2 weight decay for regularisation
- Positive class weight = 8.0 to handle ~10% minority class
- 80/20 reproducible train/val split

### 4. Symbolic Clinical Rules
Vectorised Prolog rules applied as pandas operations:
```prolog
renal_risk(P)     :- creatinine(P, Cr), Cr > 2.0.
sepsis_risk(P)    :- wbc(P, W), (W > 12.0 ; W < 4.0).
metabolic_risk(P) :- glucose(P, G), G > 200.
high_risk(P)      :- renal_risk(P), sepsis_risk(P).
high_risk(P)      :- renal_risk(P), metabolic_risk(P).
medium_risk(P)    :- sepsis_risk(P), metabolic_risk(P).
```

### 5. Hybrid Fusion & Evaluation
- 37-dim hybrid vector = 32 GNN dims + 5 Prolog rule scores
- Logistic Regression and Gradient Boosting classifiers
- 5-fold stratified cross-validation
- Metrics: F1 macro, F1 (positive class), ROC-AUC, confusion matrix
- Ablation study across GNN-only, Prolog-only, and Hybrid configurations

---

## 🔧 Requirements

```bash
pip install torch torch_geometric scikit-learn pandas numpy
```

> **Kaggle users:** Enable **Internet** in notebook settings before running. Environment setup takes ~2–3 minutes.

### Key Libraries

| Library | Version |
|---|---|
| PyTorch | ≥ 2.0 |
| PyTorch Geometric | latest |
| scikit-learn | ≥ 1.0 |
| pandas | ≥ 1.5 |
| numpy | ≥ 1.23 |

---

## 🚀 Usage

1. Download the [MIMIC-III Demo dataset](https://www.kaggle.com/datasets/atamazian/mimic-iii-clinical-dataset-demo) from Kaggle.
2. Update the `BASE` path in **Cell 2** to point to your dataset location.
3. Run all cells sequentially (Cell 1 → Cell 8).

The notebook is self-contained and designed to run on **Kaggle Notebooks** (GPU optional).

---

## 📊 Results

Evaluation is performed on a held-out 20% test set with stratified splitting. The ablation study compares:

- **GNN only** (32 features)
- **Prolog only** (5 rule scores)
- **Hybrid** (GNN + Prolog, 37 features) ✅ Best performance

Metrics reported: F1 macro, F1 (positive / expired class), ROC-AUC.

---

## 🔮 Potential Extensions

| Extension | Description |
|---|---|
| **Temporal GNN** | Replace static snapshot with time-aware message passing (e.g., T-GCN) |
| **Full MIMIC-III** | Scale to ~46k admissions using PyG's `NeighborLoader` for mini-batch training |
| **SHAP explanations** | Quantify feature importance across the hybrid feature space |
| **Real PyProlog integration** | Use actual Prolog inference for richer, recursive rule chains |
| **Attention GNN** | Replace GCNConv with GATConv for interpretable edge weights |

---

## 👤 Author

**Nihant**

---

## ⚠️ Disclaimer

This notebook is developed for **educational and research purposes only**. It is not intended for clinical deployment or medical decision-making. MIMIC-III data access requires credentialed authorization via [PhysioNet](https://physionet.org/).
