# Random Forest Notebook Readiness Report

## Question: Is the random forest notebook ready to run on NMDC input?

**Answer: NO** - The notebook requires 3 updates before it can run on NMDC data.

## ✅ Good News: Dataset Compatibility

The NMDC complete dataset has the **exact columns** the RF notebook expects:

**NMDC columns:**
```
accession, collection_date, latitude, longitude, env_broad_scale, env_local_scale,
env_medium, google_earth_embeddings, envo_broad_scale_embedding,
envo_medium_embedding, envo_local_scale_embedding
```

**RF notebook expects:**
- `google_earth_embeddings` ✅ (present)
- `env_local_scale` ✅ (present)
- `env_broad_scale` ✅ (present - available but not used)
- `env_medium` ✅ (present - available but not used)

## ❌ Required Changes

### Change 1: Update Dataset Path (Cell 4)

**Current:**
```python
df = pd.read_csv('../data/satisfying_biosamples_normalized_with_google_embeddings_with_envo_embeddings.csv',
                 on_bad_lines='skip', engine='python')
```

**Required:**
```python
df = pd.read_csv('../data/nmdc_flattened_biosample_for_env_embeddings_202510061052_complete.csv')
```

**Impact:** Critical - without this change, notebook will use wrong dataset

---

### Change 2: Expand to All 3 MIxS Scales (Multiple Cells)

**Current:** Only predicts `env_local_scale`

**Required:** Train 3 separate Random Forest classifiers:
1. `env_broad_scale` → Biome level prediction
2. `env_local_scale` → Fine-grained prediction (current)
3. `env_medium` → Intermediate prediction

**Cells to modify:**
- Cell 5: Check for all 3 columns, not just `env_local_scale`
- Cell 9: Change target variable section to handle all 3
- Cell 12: Create 3 target vectors (y_broad, y_local, y_medium)
- Cell 16-31: Duplicate training/evaluation for each scale

**Impact:** Important - aligns with project goal of full triad prediction

---

### Change 3: Add Comparison to k-NN Baseline (New Section)

**Current:** Only evaluates RF accuracy in isolation

**Required:** Compare RF predictions to the k-NN baseline from `similarity_analysis.ipynb`

**Suggested new section:**
```python
## Compare to k-NN Baseline

# For samples where RF has low confidence, compare to k-NN prediction
# This validates whether RF is better than simple nearest-neighbor approach
```

**Impact:** Medium priority - provides scientific validation

---

## Expected Results After Update

### Sample Size Comparison
- **Previous RF work:** 10,978 samples (accidental NCBI subset)
- **NMDC dataset:** 8,121 samples (curated gold standard)

### Scale Coverage
- **Previous:** 1 classifier (local_scale only)
- **After update:** 3 classifiers (broad, local, medium)

### Success Metrics
The updated notebook should report:
1. Training/test accuracy for each of 3 scales
2. Feature importance rankings (which GE embedding dims matter most)
3. Prediction confidence distributions
4. Comparison of RF vs k-NN baseline performance
5. Per-class precision/recall for common ENVO terms

---

## Recommended Workflow

1. **Update Cell 4** (dataset path) → Run to verify data loads correctly
2. **Run current notebook as-is** on NMDC to get local_scale baseline
3. **Expand to 3 scales** → Compare which scale is most predictable
4. **Add k-NN comparison** → Validate RF provides value over simpler method

---

## Dataset Statistics (from similarity_analysis.ipynb)

```
Total rows loaded: 8121
Columns: ['accession', 'collection_date', 'latitude', 'longitude',
          'env_broad_scale', 'env_local_scale', 'env_medium',
          'google_earth_embeddings', 'envo_broad_scale_embedding',
          'envo_medium_embedding', 'envo_local_scale_embedding']

=== EMBEDDING DATA QUALITY ===
Rows with google_earth_embeddings: 8121
Rows with envo_broad_scale_embedding: 8121
Rows with envo_local_scale_embedding: 8121
Rows with envo_medium_embedding: 8121

Rows with ALL embeddings (GE + 3 ENVO): 8121
Success rate: 100.0%
```

All 8,121 samples have complete embeddings - no data quality issues.
