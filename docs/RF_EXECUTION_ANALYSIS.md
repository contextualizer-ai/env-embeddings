# Random Forest Notebook Execution Analysis

## Does the RF notebook downsample?

**Answer: NO** - The notebook processes all data provided to it.

### Data Flow

1. **Load:** Reads entire CSV file
   ```python
   df = pd.read_csv('../data/nmdc_flattened_biosample_for_env_embeddings_202510061052_complete.csv')
   ```

2. **Filter:** Only removes rows with missing values
   ```python
   df_clean = df[df['google_earth_embeddings'].notna() & df['env_local_scale'].notna()].copy()
   ```
   - NMDC dataset: 8,121 rows → 8,121 rows retained (100% success rate)
   - Previous NCBI subset: 10,978 rows → 10,978 rows retained (100% success rate)

3. **Train/Test Split:** Standard 80/20 split
   ```python
   test_size = 0.2
   X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
   ```
   - NMDC (8,121): Train = 6,497, Test = 1,624
   - Previous (10,978): Train = 8,782, Test = 2,196

4. **Training:** Uses ALL training samples
   - No random forest max_samples parameter (uses 100% by default)
   - All 100 trees see all training data (with bootstrap sampling)

### No Downsampling Found

Verified by grep search - no occurrence of:
- `.sample()` calls
- `.head()` limiting data (only used for display)
- Manual row slicing beyond train/test split
- `max_samples` parameter in RandomForestClassifier

## Execution Time Estimate

### Previous Execution (random_forest_predict_envo branch)

**Dataset:** ~10,978 NCBI samples (accidental subset)

**Timeline from git commits:**
```
035825d  2025-10-03 10:56:05  "Notebook to build and test random forest..."
         (template created)

73c7071  2025-10-03 10:58:51  "Ran notebook"
         (execution complete)
```

**Elapsed time:** ~2.75 minutes (165 seconds)

### Execution Stages

The notebook has 32 cells. Major computational steps:

1. **Cell 2:** Imports (< 5 seconds)
2. **Cell 4:** Load CSV (~5-10 seconds)
3. **Cell 7:** Parse 10,978 embeddings (~10-15 seconds)
4. **Cell 16:** **Train Random Forest** (bulk of time - estimated ~60-90 seconds)
   - 100 trees
   - max_depth=10
   - 8,782 training samples
   - 64 features
   - 553 classes
5. **Cell 18:** Predictions (~5-10 seconds)
6. **Cell 19:** 5-fold cross-validation (~30-60 seconds)
7. **Cell 20:** Classification report (~5 seconds)
8. **Cell 22:** Confusion matrix plot (~5 seconds)
9. **Remaining cells:** Analysis and visualization (~15-30 seconds)

**Total estimated:** 2-3 minutes matches the git timeline

### Expected NMDC Execution Time

**Dataset:** 8,121 samples (74% of previous size)

**Scaling factors:**
- Training samples: 6,497 vs 8,782 (74% of previous)
- Number of classes: Unknown (likely similar ~500-600 ENVO terms)
- Features: Same (64 dimensions)
- Trees: Same (100)
- Depth: Same (10)

**Random Forest training scales roughly O(n * log(n))** for sample count

**Estimated time:**
- Previous: ~165 seconds for 8,782 training samples
- NMDC: ~120-140 seconds for 6,497 training samples
- **Total notebook runtime: ~2-2.5 minutes**

### Timing Breakdown (NMDC estimate)

| **Cell/Stage** | **Estimated Time** |
|----------------|-------------------|
| Imports | 2-5 sec |
| Load CSV | 5-10 sec |
| Parse embeddings | 8-12 sec |
| **Train RF** | **60-80 sec** |
| Predictions | 5-8 sec |
| Cross-validation | 25-40 sec |
| Reports/plots | 10-20 sec |
| **TOTAL** | **~2-3 min** |

### Bottlenecks

1. **Random Forest training** - Dominant cost (~50-60% of runtime)
2. **5-fold CV** - Second largest (~20-30% of runtime)
3. **Embedding parsing** - Minor (~5-10% of runtime)

### Scaling to Larger Datasets

**NCBI complete (382,955 samples):**
- 47x more samples than NMDC
- Training samples: ~306,364 (80% split)
- Estimated RF training time: ~30-60 minutes (assuming linear scaling)
- Estimated CV time: ~15-30 minutes
- **Total notebook runtime: ~45-90 minutes** (rough estimate)

**Mitigation strategies for large datasets:**
1. Reduce n_estimators (100 → 50)
2. Limit max_samples (e.g., 50% of data per tree)
3. Skip cross-validation (just do train/test split)
4. Subsample for initial exploration, then full run for final model

## Jupyter Metadata

**Note:** The executed notebook does NOT contain cell-level execution timing metadata. Standard Jupyter notebooks don't record this by default unless:
- Using Jupyter Lab with timing extension
- Using `%%time` magic in cells
- Using nbconvert with timing options

The git commit timeline provides only an upper bound on total execution time.

## Recommendations

### For NMDC (8,121 samples)
- ✅ Current configuration is fine
- Runtime: ~2-3 minutes
- No optimization needed

### For NCBI complete (382,955 samples)
- ⚠️ Current configuration may be slow (~45-90 min)
- Consider optimizations:
  - `n_estimators=50` (faster, still good performance)
  - `max_samples=0.5` (each tree sees 50% of data)
  - Skip CV for initial runs
  - Use `verbose=2` to monitor progress

### For Development/Iteration
- Always test on NMDC first (fast)
- Only run NCBI complete for final results
- Consider stratified sampling (e.g., 10% of NCBI) for prototyping
