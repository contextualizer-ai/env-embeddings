# Scientific Analysis: Methodological Concerns in similarity_analysis.ipynb

## Summary of Critical Issues

The current notebook shows high correlation coefficients (r=0.811 for broad_scale), but several methodological concerns suggest these results may be inflated or unreliable:

1. **Self-pairs inflation** (most critical)
2. **Sample size adequacy**
3. **Sampling methodology**
4. **Statistical independence**

---

## 1. Self-Pairs Problem: Inflated Correlations from Identical Samples

### The Issue

Looking at the output from cell-15 and cell-17, we see **many pairs with similarity = 1.0 for BOTH Google Earth AND all ENVO types**:

```
High similarity in both (GE > 0.9, ENVO broad_scale > 0.9): 212 pairs
```

**Out of 1000 pairs, 212 have near-perfect agreement (>0.9 on both metrics).**

From cell-17 output, we can see why:

```
Pair: SAMD00115491 vs SAMD00115487
GE Similarity: 1.000
ENVO broad_scale: 1.000
Sample 1 - Lat/Lon: (35.7128, 139.7619), Date: 2018-02-11
Sample 2 - Lat/Lon: (35.7128, 139.7619), Date: 2018-02-11
Sample 1 ENVO: ENVO:00002030 | ENVO:00002006 | ENVO:00000241
Sample 2 ENVO: ENVO:00002030 | ENVO:00002006 | ENVO:00000241
```

**These are essentially the same sample!** Same location, same date, same ENVO terms → 1.0 similarity on both axes.

### Why This Inflates Correlation

When you have many (1,1) points in a correlation analysis:
- Pearson correlation is **extremely sensitive** to these perfect-agreement points
- They create an artificial "anchor" at (1,1) that pulls the correlation line upward
- This can make a weak/moderate correlation appear strong

### Evidence in the Data

From the current run with 246 samples:
- **212/1000 pairs (21%)** have GE > 0.9 AND ENVO broad > 0.9
- Many of these are likely **self-pairs** (same location, same ENVO terms)
- The dataset has many samples from **same locations** (e.g., (35.7128, 139.7619) appears multiple times)

### The Solution

**Exclude self-pairs and near-duplicates:**

1. **Remove identical samples**: Don't compare samples with identical coordinates + ENVO terms
2. **Apply similarity threshold**: Exclude pairs where EITHER metric > 0.95 (likely duplicates)
3. **Re-calculate correlations** on the cleaned dataset

---

## 2. Sample Size: Is 1000 Pairs Enough?

### Current Situation

- **246 samples** total
- **1000 random pairs** generated
- Maximum possible unique pairs: C(246, 2) = **30,135 pairs**
- Current sampling: **1000/30,135 = 3.3%** of all possible pairs

### Guidelines for Pair Sampling

From statistical literature on correlation analysis:

**Power Analysis for Correlation:**
- To detect r=0.8 with 80% power and α=0.05: **n ≥ 12 pairs**
- To detect r=0.5 with 80% power: **n ≥ 29 pairs**
- To detect r=0.3 with 80% power: **n ≥ 84 pairs**

So **1000 pairs is MORE than sufficient** for statistical power.

**But...**

### The Real Question: Representative Sampling

With 246 samples and 1000 pairs:
- Average sample appears in: 1000 × 2 / 246 ≈ **8 pairs**
- Some samples may appear 0 times, others 20+ times (random variation)
- This creates **dependency** between pairs (not truly independent observations)

### Recommendations

**Option 1: Sample ALL pairs** (computational cost)
- 30,135 pairs is not computationally prohibitive
- Ensures complete coverage
- Removes sampling bias

**Option 2: Stratified random sampling**
- Ensure each sample appears in at least N pairs (e.g., 5-10)
- Better coverage than pure random sampling
- Still computationally efficient

**Option 3: Increase to 5000-10000 pairs**
- Covers ~17-33% of all possible pairs
- More stable estimates
- Since computation is fast (you mentioned "doesn't take too much time"), this is easy to do

---

## 3. Sampling Methodology: Random Pairs with Replacement

### Current Implementation (Cell 8)

```python
n_pairs = 1000
for i in range(n_pairs):
    idx1, idx2 = random.sample(range(n_samples), 2)
    pairs.append((idx1, idx2))
```

**This is sampling WITHOUT replacement within each pair** (idx1 ≠ idx2), but **WITH replacement across pairs** (same pair can be selected multiple times).

### Problems

1. **Duplicate pairs**: (A, B) could be selected multiple times
2. **Asymmetric pairs**: Both (A, B) and (B, A) could be selected (though similarity is symmetric, this wastes samples)
3. **Non-uniform coverage**: Some pairs over-sampled, others never sampled

### Better Approach

**For small datasets (like 246 samples):**

```python
# Generate ALL unique pairs
from itertools import combinations
all_pairs = list(combinations(range(n_samples), 2))

# Optional: Sample from all_pairs if needed
if len(all_pairs) > max_pairs:
    pairs = random.sample(all_pairs, max_pairs)
else:
    pairs = all_pairs
```

**Advantages:**
- No duplicate pairs
- Guaranteed unique coverage
- More statistically rigorous

---

## 4. Dependency Between Pairs

### The Problem

Pairs are **not independent observations** when samples are reused:

Example:
- Pair 1: (A, B) → similarity = 0.8
- Pair 2: (A, C) → similarity = 0.6
- Pair 3: (B, C) → similarity = 0.7

All three pairs share samples, so they're not independent. This violates the **independence assumption** of correlation tests (Pearson, Spearman).

### Impact on p-values

The reported p-values (e.g., p=6.81e-235) assume **independent observations**. With dependent pairs:
- p-values are **too small** (overconfident)
- Standard errors are **underestimated**
- Confidence intervals are **too narrow**

### Correction Approaches

**Option 1: Permutation testing**
- Shuffle one embedding type, recalculate correlation, repeat 10,000 times
- Empirical p-value from permutation distribution
- Doesn't assume independence

**Option 2: Bootstrap confidence intervals**
- Resample samples (not pairs), generate new pairs, calculate correlation
- Repeat 10,000 times
- 95% CI from bootstrap distribution

**Option 3: Report effect size only**
- Don't rely on p-values
- Focus on correlation magnitude and practical significance

---

## 5. Proposed Sensitivity Analysis

### Test 1: Exclude High-Similarity Pairs

Remove pairs where **either** GE or ENVO similarity > 0.95:

```python
# Exclude likely duplicates
pairs_filtered = pairs_df[
    (pairs_df['ge_similarity'] < 0.95) &
    (pairs_df['envo_broad_similarity'] < 0.95)
]

# Recalculate correlation
pearson_filtered, p_filtered = pearsonr(
    pairs_filtered['ge_similarity'],
    pairs_filtered['envo_broad_similarity']
)
```

**Expected outcome:**
- If correlation drops significantly (e.g., 0.811 → 0.4), the result is driven by duplicates
- If correlation stays strong (0.811 → 0.7+), the relationship is robust

### Test 2: Exclude Perfect Matches

Remove pairs where **both** GE AND ENVO similarity = 1.0:

```python
pairs_no_perfect = pairs_df[
    ~((pairs_df['ge_similarity'] == 1.0) &
      (pairs_df['envo_broad_similarity'] == 1.0))
]
```

### Test 3: Increase Sample Size

Run with **10,000 pairs** or **ALL possible pairs**:

```python
# Generate all unique pairs
from itertools import combinations
all_pairs = list(combinations(range(len(df_clean)), 2))
print(f"Total unique pairs: {len(all_pairs)}")
```

Check if correlation coefficient stabilizes or changes.

### Test 4: Stratified Sampling

Ensure diverse pairs by sampling across environmental categories:

```python
# Group by ENVO broad_scale term
groups = df_clean.groupby('env_broad_scale')

# Sample pairs within and across groups
within_group_pairs = []  # Same ENVO term
across_group_pairs = []  # Different ENVO terms
```

---

## 6. Additional Statistical Concerns

### Multiple Testing

The notebook tests **3 ENVO types** (broad, medium, local) against GE.
- With no correction, we expect 1/20 false positives at α=0.05
- Should apply **Bonferroni correction**: α_corrected = 0.05/3 = 0.0167
- Or report False Discovery Rate (FDR) adjusted p-values

### Distribution of Similarities

From cell-11 output:
```
Google Earth Embeddings:
  Mean: 0.381
  Range: [-0.088, 1.000]

ENVO broad_scale:
  Mean: 0.636
  Range: [0.210, 1.000]
```

**GE similarities can be negative** (cosine similarity range: [-1, 1])
**ENVO similarities are all positive** (OpenAI embeddings are typically [0, 1])

This **asymmetry** might affect correlation:
- Pearson correlation assumes **linear relationship**
- If relationship is non-linear, Pearson may be misleading
- Spearman (rank-based) is more robust

**Observation:** Spearman is consistently lower (0.677 vs 0.811 for broad_scale)
- Suggests relationship may not be perfectly linear
- Or there are outliers influencing Pearson

---

## 7. Recommendations

### Immediate Actions (Easy Wins)

1. **Add sensitivity analysis cell:**
   - Exclude pairs with similarity > 0.95 on either metric
   - Exclude perfect (1.0, 1.0) pairs
   - Report correlations before/after filtering

2. **Increase to ALL pairs or 10,000 pairs:**
   - You said computation is fast
   - This is scientifically more rigorous
   - 30,135 pairs is not computationally expensive

3. **Add robustness checks:**
   - Bootstrap confidence intervals (1000 iterations)
   - Permutation test for p-values
   - Report effect sizes with CIs, not just p-values

### Medium-Term Improvements

4. **Stratified analysis:**
   - Separate correlations by ENVO term frequency
   - Do rare ENVO terms show different patterns?

5. **Visualize outliers:**
   - Identify pairs driving the correlation
   - Are high-correlation points real or duplicates?

6. **Multiple testing correction:**
   - Bonferroni or FDR correction for 3 ENVO types

### Long-Term (Requires More Data)

7. **Process more samples:**
   - Current: 246 samples
   - You mentioned "we can easily provide more now"
   - Target: 1000+ samples for robust analysis

8. **Cross-validation:**
   - Split data into train/test sets
   - Check if correlation generalizes

---

## 8. Concrete Next Steps

### Create New Notebook Cell: Sensitivity Analysis

```python
print("=== SENSITIVITY ANALYSIS ===")

# Original correlation
print(f"\nOriginal (all {len(pairs_df)} pairs):")
print(f"  Pearson r = {pearson_broad:.3f}")

# Test 1: Remove high-similarity pairs
pairs_filtered = pairs_df[
    (pairs_df['ge_similarity'] < 0.95) &
    (pairs_df['envo_broad_similarity'] < 0.95)
]
r_filtered, p_filtered = pearsonr(
    pairs_filtered['ge_similarity'],
    pairs_filtered['envo_broad_similarity']
)
print(f"\nAfter removing similarity > 0.95 ({len(pairs_filtered)} pairs):")
print(f"  Pearson r = {r_filtered:.3f}")
print(f"  Change: {r_filtered - pearson_broad:.3f}")

# Test 2: Remove perfect matches
pairs_no_perfect = pairs_df[
    ~((pairs_df['ge_similarity'] == 1.0) &
      (pairs_df['envo_broad_similarity'] == 1.0))
]
r_no_perfect, p_no_perfect = pearsonr(
    pairs_no_perfect['ge_similarity'],
    pairs_no_perfect['envo_broad_similarity']
)
print(f"\nAfter removing (1.0, 1.0) pairs ({len(pairs_no_perfect)} pairs):")
print(f"  Pearson r = {r_no_perfect:.3f}")
print(f"  Change: {r_no_perfect - pearson_broad:.3f}")

# Test 3: Bootstrap confidence intervals
from scipy.stats import bootstrap
data = (pairs_df['ge_similarity'].values, pairs_df['envo_broad_similarity'].values)
def correlation_statistic(x, y):
    return pearsonr(x, y)[0]

# This requires scipy >= 1.7
# res = bootstrap(data, correlation_statistic, n_resamples=1000, method='percentile')
# print(f"\n95% Bootstrap CI: [{res.confidence_interval.low:.3f}, {res.confidence_interval.high:.3f}]")
```

### Modify Cell 8: Sample ALL Pairs

```python
from itertools import combinations

# Generate ALL unique pairs instead of random sample
print(f"Generating all unique pairs from {n_samples} samples...")
all_possible_pairs = list(combinations(range(n_samples), 2))
print(f"Total unique pairs: {len(all_possible_pairs)}")

# Use all pairs (computation is fast)
pairs = all_possible_pairs

# Or limit if needed (but 30K pairs should be fast)
# max_pairs = 10000
# if len(all_possible_pairs) > max_pairs:
#     pairs = random.sample(all_possible_pairs, max_pairs)
# else:
#     pairs = all_possible_pairs
```

---

## 9. Expected Outcomes

### If correlation remains strong after filtering (r > 0.7):
✅ The relationship is **real and robust**
✅ Geographic and ontological similarities truly correlate
✅ Results are publishable with proper caveats

### If correlation drops substantially (r < 0.5):
❌ Original result was **driven by duplicates/self-pairs**
❌ Need to investigate why so many near-identical samples
❌ May need to deduplicate dataset before analysis

### Most Likely Scenario:
- Correlation drops moderately (0.811 → 0.6-0.7)
- Still significant and meaningful
- More honest assessment of relationship strength

---

## 10. Summary

**Critical Flaw:** 21% of pairs are near-perfect matches (>0.9, >0.9), likely inflating correlation

**Quick Fix:**
1. Filter out high-similarity pairs (>0.95)
2. Use ALL 30,135 pairs instead of 1000 random pairs
3. Report correlation with and without filtering

**Scientific Rigor:**
- Current p-values are likely too optimistic (dependency issue)
- Need bootstrap CIs or permutation tests
- Should report effect sizes, not just significance

**Data Size:**
- 1000 pairs is statistically sufficient
- But ALL pairs is better (more stable, no sampling bias)
- Processing more samples (1000+ vs current 246) would strengthen conclusions

**Bottom Line:** The high correlation (r=0.811) is **suspicious** until we verify it's not driven by duplicate/near-duplicate samples. The sensitivity analysis will reveal the truth.
