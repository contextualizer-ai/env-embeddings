# Notebook Guide: Environmental Embeddings Analysis

## Overview

This repository contains Jupyter notebooks for analyzing environmental sample metadata using satellite imagery embeddings and environmental ontology (ENVO) embeddings. The notebooks can be run **independently** - they do not need to be executed in a specific order.

## Notebooks

### 1. `similarity_analysis.ipynb` (803 KB, executed with outputs)

**Purpose:** Cross-validate environmental annotations with satellite imagery

**Research Question:** Do environmental samples with similar satellite imagery have similar ENVO annotations? Can we use satellite data to detect metadata errors?

**Dataset:** NMDC complete (8,121 samples)
- `nmdc_flattened_biosample_for_env_embeddings_202510061052_complete.csv`

**Key Analyses:**
1. **Pairwise Similarity Analysis**
   - Compares 1 million sample pairs
   - Computes cosine similarity between:
     - Google Earth embeddings (64-dim satellite imagery)
     - ENVO embeddings (1536-dim for each of 3 scales)

2. **Degenerate Pair Filtering**
   - Removes near-duplicate pairs (similarity > 0.95)
   - Tests correlation robustness
   - 679,048 pairs retained after filtering

3. **Multi-Scale ENVO Comparison**
   - Tests all 3 MIxS scales:
     - `env_broad_scale` (biome level)
     - `env_local_scale` (fine-grained) → **BEST: r=0.307**
     - `env_medium` (intermediate)
   - Tests combined embeddings (concatenated & weighted average)

4. **Outlier Detection**
   - Finds pairs with high GE similarity but low ENVO similarity
   - Identifies "chronic outliers" appearing in many disagreement pairs
   - Flags metadata quality issues

5. **k-NN Triad Prediction**
   - Predicts ENVO triads from k=5 GE-similar samples
   - Uses majority voting
   - Serves as baseline for RF comparison

**Key Findings:**
- Local scale shows strongest correlation (r=0.307, ρ=0.214)
- Broad scale shows weak negative correlation (r=-0.066)
- Degenerate filtering reduced correlation by Δ=-0.253 (indicates inflated original results)

**Use Cases:**
- Metadata quality control
- Understanding GE ↔ ENVO relationship
- Baseline k-NN predictions for comparison

---

### 2. `random_forest_envo_prediction.ipynb` (16 KB, template without outputs)

**Purpose:** Train Random Forest classifiers to predict ENVO terms from satellite embeddings

**Research Question:** Can geographic/satellite imagery data predict environmental classification labels?

**Dataset:**
- **Currently configured:** NMDC complete (8,121 samples) ✅ UPDATED
- **Previously used:** Accidental NCBI subset (~10,978 samples) - see executed version

**Key Analyses:**
1. **Binary Classification** (currently only `env_local_scale`)
   - Features: 64-dim Google Earth embeddings
   - Target: ENVO local scale term (553 classes)
   - Model: RandomForestClassifier (100 trees, max_depth=10)

2. **Train/Test Split**
   - 80/20 split
   - Stratification if possible (min 2 samples per class)

3. **Model Evaluation**
   - Training vs test accuracy (overfitting check)
   - 5-fold cross-validation
   - Classification report (precision/recall per class)
   - Confusion matrix

4. **Feature Importance**
   - Identifies which GE embedding dimensions matter most
   - Ranks top 20 features

5. **Prediction Confidence Analysis**
   - Examines max probability distributions
   - Shows low-confidence vs high-confidence predictions

**Current Limitations:**
- ⚠️ Only predicts `env_local_scale` (should expand to all 3 scales)
- ⚠️ No comparison to k-NN baseline from `similarity_analysis.ipynb`

**Expected Performance** (based on executed version with ~10,978 samples):
- Training accuracy: ~63.7%
- Test accuracy: ~53.1%
- Mean prediction confidence: ~20.3%
- Finding: "MODERATE predictive power" with moderate overfitting

**Use Cases:**
- Predict missing ENVO terms from satellite imagery
- Compare supervised (RF) vs unsupervised (k-NN) methods
- Understand which environmental scales are most predictable

**Planned Improvements:**
1. Expand to all 3 MIxS scales (broad, local, medium)
2. Add k-NN baseline comparison
3. Test on larger NCBI dataset (382,955 samples)

---

### 3. `random_forest_envo_prediction_executed.ipynb` (626 KB, with outputs)

**Purpose:** Historical record of RF classifier trained on accidental NCBI subset

**Dataset:** `satisfying_biosamples_normalized_with_google_embeddings_with_envo_embeddings.csv`
- 10,978 samples (accidental NCBI subset from earlier pipeline)
- **NOT the complete datasets** now available

**Results:**
- Test accuracy: 53.1%
- 553 unique ENVO local_scale classes
- Most common class: ENVO:00001998 (464 samples, 4.2%)

**Status:** Reference only - shows what was possible with previous dataset

---

## Conceptual Relationships

### Independent Approaches to Same Problem

```
Research Goal: Understand relationship between satellite imagery and ENVO annotations
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
         Correlation Analysis          Classification Approach
      (similarity_analysis.ipynb)  (random_forest_envo_prediction.ipynb)
                    │                           │
            ┌───────┴───────┐           ┌──────┴──────┐
            │               │           │             │
      Pairwise Cosine   k-NN Baseline   RF Supervised
      Similarity        Prediction      Learning
      (unsupervised)    (lazy learning) (eager learning)
```

### Complementary Insights

| **Aspect** | **Similarity Analysis** | **Random Forest** |
|------------|------------------------|-------------------|
| **Method** | Pairwise cosine similarity | Supervised classification |
| **Question** | "Do similar images have similar labels?" | "Can we predict labels from images?" |
| **Output** | Correlation coefficient (r) | Classification accuracy (%) |
| **Strengths** | - Tests all 3 ENVO scales<br>- Identifies outliers<br>- k-NN baseline | - Predicts for unlabeled samples<br>- Feature importance<br>- Handles multi-class |
| **Use Case** | Metadata quality control | Missing metadata prediction |

### Recommended Workflow

**For metadata quality control:**
1. Run `similarity_analysis.ipynb` first
2. Identify outliers and chronic offenders
3. Use k-NN predictions as first-pass suggestions

**For missing metadata prediction:**
1. Run `random_forest_envo_prediction.ipynb` to train models
2. Compare RF accuracy to k-NN baseline from similarity analysis
3. Use whichever method has higher confidence for predictions

**For research/publication:**
1. Run both notebooks on same dataset (NMDC complete)
2. Report correlation results AND classification accuracy
3. Show RF provides value over simple k-NN approach

---

## Dataset Files

### Current Complete Datasets (branch: 46-complete-ncbi-dataset)

```
data/
  nmdc_flattened_biosample_for_env_embeddings_202510061052_complete.csv
    → 8,121 samples (100% embedding coverage)
    → Curated gold standard
    → Used by: similarity_analysis.ipynb, random_forest_envo_prediction.ipynb

  ncbi_flattened_biosamples_for_env_embeddings_202510061108_normalized_complete.csv
    → 382,955 samples (88.6% after coordinate dedup)
    → Full NCBI BioSample
    → Future: Scale RF training to this

  gold_flattened_biosample_for_env_embeddings_202510061143_normalized_complete.csv
    → 10,401 samples
    → GOLD database samples
    → Future: Independent test set
```

### Legacy Dataset (historical reference)

```
data/satisfying_biosamples_normalized_with_google_embeddings_with_envo_embeddings.csv
  → ~10,978 samples (accidental NCBI subset)
  → Used by: random_forest_envo_prediction_executed.ipynb
  → No longer recommended for new analyses
```

---

## Required Columns

All notebooks expect these columns:

| **Column** | **Description** | **Type** |
|------------|----------------|----------|
| `accession` | Sample ID (SAMD*, SAMN*) | string |
| `collection_date` | ISO date | string |
| `latitude` | Decimal degrees | float |
| `longitude` | Decimal degrees | float |
| `env_broad_scale` | ENVO ID (biome) | string |
| `env_local_scale` | ENVO ID (fine-grained) | string |
| `env_medium` | ENVO ID (intermediate) | string |
| `google_earth_embeddings` | 64-dim list (as string) | string |
| `envo_broad_scale_embedding` | 1536-dim list (as string) | string |
| `envo_local_scale_embedding` | 1536-dim list (as string) | string |
| `envo_medium_embedding` | 1536-dim list (as string) | string |

---

## Execution Order

**Short answer:** No specific order required - notebooks are independent.

**Recommended for first-time users:**
1. `similarity_analysis.ipynb` - Start here to understand the data and GE↔ENVO relationship
2. `random_forest_envo_prediction.ipynb` - Then try supervised learning approach

**For iteration/development:**
- Modify and re-run either notebook independently
- Both read from same data files, no dependencies between them

---

## Next Steps (Issue #49)

Current branch `49-nmdc-rf-envo-prediction` is updating RF notebook to:
1. ✅ Use NMDC complete dataset (done)
2. ⬜ Expand to all 3 MIxS scales
3. ⬜ Add k-NN baseline comparison
4. ⬜ Validate against similarity_analysis findings

See `RF_NOTEBOOK_READINESS.md` for detailed change requirements.
