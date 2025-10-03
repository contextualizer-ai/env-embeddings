# Analysis Findings: Satellite Imagery vs ENVO Annotations

## Executive Summary

After rigorous filtering of near-duplicate pairs, the correlation between Google Earth satellite embeddings and ENVO (Environmental Ontology) annotations is **weak** (r = 0.167). This negative result is scientifically valuable and suggests that satellite imagery and semantic annotations capture fundamentally different aspects of environmental context.

---

## 🔴 Critical Finding: Correlation Collapse

### The Numbers

| Metric | Before Filtering | After Filtering | Change |
|--------|-----------------|-----------------|--------|
| **Pearson (broad_scale)** | 0.789 | 0.114 | **-0.675** |
| **Pearson (local_scale)** | 0.647 | 0.167 | **-0.480** |
| **Pearson (medium)** | 0.756 | -0.054 | **-0.810** |

### What Happened

**31.7% of pairs were degenerate** (similarity > 0.95 on at least one metric):
- 2,080 pairs with GE similarity > 0.95 (20.8%)
- 2,381 pairs with ENVO broad_scale > 0.95 (23.8%)
- 2,983 pairs with ENVO medium > 0.95 (29.8%)

These near-duplicate pairs created artificial correlation through the (1,1) anchor point problem:
- Samples from same location → GE = 1.0
- Samples with same ENVO terms → ENVO = 1.0
- Many such pairs → inflates correlation coefficient

**After filtering:** Only 6,826 pairs remain from 10,000 original pairs.

### Why This Matters

This demonstrates:
1. **Methodological rigor is critical** in similarity studies
2. **Degenerate pair filtering is essential** to avoid false positives
3. **The original hypothesis (strong correlation) was wrong** - and that's valuable science

---

## 📊 What the Data Actually Shows

### Best Correlations (After Filtering)

**Winner: Local scale (fine-grained environmental context)**
- Pearson r = 0.167 (weak linear)
- Spearman ρ = 0.336 (moderate rank)
- Local scale = specific features like "agricultural field", "coastal zone", "urban"

**Runner-up: Concatenated (all three scales combined)**
- Pearson r = 0.161
- Spearman ρ = 0.412 (**highest of all methods**)
- 4608-dimensional vector (1536 × 3)

### Key Observations

**1. Spearman >> Pearson everywhere**
```
Local scale:  Spearman (0.336) vs Pearson (0.167)
Combined:     Spearman (0.412) vs Pearson (0.161)
```
→ **Non-linear relationship** between satellite and ENVO similarity

**2. Medium scale has negative Pearson correlation**
- Pearson r = -0.054 (statistically significant, p = 7.25e-06)
- Spearman ρ = 0.125 (still positive)
- Medium = intermediate features like "soil", "sea water", "sediment"
- **Requires investigation:** Why does linear correlation flip negative?

**3. Concatenation performs well on Spearman**
- Highest Spearman (0.412) but not highest Pearson (0.161)
- Suggests combining scales helps for rank-based comparison
- But doesn't improve linear correlation

### Distribution Statistics (Filtered Data)

**Google Earth similarity:**
- Mean: 0.203
- Std: 0.153
- Range: [-0.088, 0.950]

**ENVO similarities:**
| Scale | Mean | Std | Range |
|-------|------|-----|-------|
| Broad | 0.527 | 0.105 | [0.210, 0.950] |
| Local | 0.311 | 0.080 | [0.164, 0.950] |
| Medium | 0.348 | 0.062 | [0.134, 0.950] |

**Key insight:** ENVO similarities are much higher and more constrained than GE similarities.

---

## 🚧 Dataset Limitations

### 1. Sample Size Too Small

- **246 samples** total
- **6,826 valid pairs** after filtering
- Maximum possible pairs: 30,135
- Only sampling 22.6% of possible space

**Consequence:** May not capture full diversity of environmental contexts.

### 2. Geographic Clustering

**Evidence from outlier analysis:**
```
Top 10 chronic outliers:
- 4 samples from (35.3202, 139.6500) - same location
- 3 samples from (35.7000, 139.5000) - same location
- Most samples from Japan (35°N, 139°E region)
```

**Consequence:**
- Limited geographic diversity
- Potential regional bias in annotations
- Same-location samples inflate pair counts

### 3. Technical Replicates

Many samples are:
- Same location
- Same date
- Same ENVO terms
- **Different accessions** (technical replicates, time series, depth profiles)

**Example from data:**
```
SAMD00093579, SAMD00093580, SAMD00093581, SAMD00093583, SAMD00093585
All from: (35.3202, 139.6500), Date: 2017-06-13
All have: ENVO:01000008 | ENVO:00002123 | ENVO:01000157
```

**Consequence:** Creates many near-identical pairs even after degenerate filtering.

### 4. Limited Biome Diversity

**Missing or underrepresented:**
- Marine environments
- Desert biomes
- Arctic/Antarctic
- Agricultural diversity
- Urban diversity

**Consequence:** Results may not generalize to all environmental contexts.

---

## 🎯 Where to Go From Here

### Option 1: Scale Up (Recommended First Step)

**Goal:** Test if weak correlation is dataset-specific or fundamental.

**Action plan:**
```python
# Process 1,000-5,000 more samples with:
- Geographic diversity (all continents, latitudes)
- Biome diversity (marine, terrestrial, urban, agricultural, extreme)
- No technical replicates (deduplicate by location+date)
- Balanced representation across ENVO terms
```

**What we'll learn:**
- Does correlation improve with scale?
- Is the dataset bias or true signal?
- Which biomes/regions drive correlation?

**Expected outcome:**
- If r stays weak (< 0.3) → correlation is real (weak)
- If r improves (> 0.5) → current dataset is biased
- If r varies by biome → need stratified analysis

---

### Option 2: Pivot to Classification/Prediction

**Reframe the question:**
- From: "How correlated are they?" (weak correlation)
- To: "Can we predict ENVO from satellite?" (measure accuracy)

**Approach: Supervised Learning**

```python
# Train a model
X = google_earth_embeddings  # 64-dim
y = envo_triads  # (broad, local, medium)

# Multi-output classification
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
accuracy_broad = accuracy_score(y_test_broad, y_pred_broad)
accuracy_local = accuracy_score(y_test_local, y_pred_local)
accuracy_medium = accuracy_score(y_test_medium, y_pred_medium)
```

**Why this is better:**
- Correlation measures linear relationship
- Classification measures **predictive power**
- Can handle non-linear patterns (we saw Spearman > Pearson)
- More interpretable for practical applications

**Baseline to beat:** Random chance
- If 10 unique broad_scale terms → random = 10% accuracy
- If model achieves 40% → 4× better than random
- Even 40% accuracy is useful for flagging suspect metadata

**Next level: k-NN approach**
- We already prototyped this in cell-19
- Find k nearest neighbors by GE similarity
- Predict ENVO triad by majority vote
- Measure accuracy, precision, recall

---

### Option 3: Focus on Outliers (Quality Control Tool)

**Forget correlation strength, use disagreement as signal.**

**High GE + Low ENVO = metadata quality issue**

**Current results:**
- 16 outlier pairs (GE > 0.8, ENVO < 0.3)
- 0.2% of filtered pairs
- 14 unique samples involved

**Why so few outliers?**
- Dataset is too small
- Geographic clustering means most samples are genuinely different locations
- Need more data to find systematic errors

**Scaling up this approach:**

```python
# With 10,000 samples
max_pairs = 10,000 * 9,999 / 2 = 49,995,000 pairs

# Even 0.1% outliers = 49,995 suspicious pairs
# Much better signal for quality control
```

**Practical application:**
1. **Automated flagging:** Submit new biosample → check GE vs ENVO
2. **Curator review:** Present top 100 disagreements per month
3. **Correction suggestions:** Use k-NN to suggest alternative triads
4. **Training data curation:** Identify high-quality samples (high agreement)

---

### Option 4: Investigate Technical Details

#### A. Why is medium scale negative?

**Observation:** Pearson r = -0.054, but Spearman ρ = +0.125

**Hypotheses:**
1. **Outliers:** A few extreme pairs flip the linear trend
2. **Non-monotonic relationship:** Medium similarity increases then decreases with GE
3. **Embedding quality:** Medium ENVO terms are less well-embedded
4. **Biological reality:** Medium features (soil, water) vary independently of satellite view

**Investigation needed:**
- Scatter plot of GE vs medium similarity (look for patterns)
- Stratify by medium term (is it specific terms driving negative correlation?)
- Compare medium embedding quality to broad/local

#### B. Why does Spearman outperform Pearson?

**Everywhere we see:** Spearman > Pearson (often 2×)

**Possible explanations:**
1. **Outliers:** Linear correlation is sensitive, rank is robust
2. **Non-linear relationship:** Similarity might be exponential/logarithmic
3. **Ceiling effects:** ENVO similarities are bounded [0, 1], GE can be negative
4. **Monotonic but not linear:** Rank order is preserved but relationship curves

**Test:**
- Try log-transformed similarities
- Try different embedding models
- Look at scatter plots for curvature

#### C. Try Different Embeddings

**Google Earth variations:**
```python
# Different zoom levels
zoom_12 = 10m resolution (current)
zoom_11 = 20m resolution (broader context)
zoom_13 = 5m resolution (finer detail)

# Temporal aggregates
median_2015_2020 = multi-year composite (reduce seasonal variation)
seasonal = separate summer/winter embeddings
```

**ENVO variations:**
```python
# Different embedding models
text-embedding-3-small (current, 1536-dim)
text-embedding-3-large (3072-dim, more expressive)

# Raw term matching
term_overlap = Jaccard similarity on ENVO IDs
semantic_similarity = from ENVO ontology structure (not text)
```

---

### Option 5: Write It Up (Publishable Negative Result)

**Title:** "Weak Correlation Between Satellite Imagery and Ontological Annotations Highlights Complementary Environmental Perspectives"

**Abstract:**
> Environmental metadata curation relies on manual annotation using standardized ontologies like ENVO. We investigated whether satellite imagery could serve as an objective proxy for ENVO annotations by comparing Google Earth embeddings to ENVO text embeddings across 246 environmental samples. Initial analysis suggested strong correlation (r = 0.789), but rigorous filtering of near-duplicate pairs revealed this was an artifact. After removing degenerate pairs, correlation was weak (r = 0.167 for best-performing local scale). Spearman correlations consistently outperformed Pearson (ρ = 0.336 vs r = 0.167), suggesting non-linear relationships. These findings demonstrate: (1) the critical importance of degenerate pair filtering in similarity studies, (2) satellite and semantic views capture complementary rather than redundant information, and (3) potential for satellite imagery in metadata quality control through outlier detection rather than direct correlation. We discuss dataset limitations and propose supervised learning approaches for future work.

**Contributions:**
1. **Methodological:** Demonstrates degenerate pair problem in similarity studies
2. **Negative result:** Weak correlation is informative (not just absence of positive result)
3. **Practical:** k-NN prediction prototype for metadata suggestion
4. **Dataset:** 246 samples with dual embeddings (satellite + ENVO)

**Sections:**
1. Introduction: Metadata quality problem in biology
2. Methods: Dual embedding approach, degenerate filtering
3. Results: Correlation collapse after filtering
4. Discussion: Why weak correlation matters, alternative approaches
5. Conclusion: Complementary perspectives, future directions

---

## 💡 Recommended Path Forward

### Phase 1: Validate with More Data (2-4 weeks)

**Objective:** Is this dataset-specific or fundamental?

**Tasks:**
1. Process 1,000 more samples (diverse geography/biomes)
   - **Note:** Pipeline now uses random sampling instead of sequential to avoid bias
   - `df.sample(n=max_rows, random_state=42)` ensures representative selection
2. Re-run correlation analysis with filtering
3. Compare results to current dataset

**Decision point:**
- If r > 0.5 → dataset bias, continue scaling
- If r < 0.3 → weak correlation is real, pivot to classification

### Phase 2: Build Prediction Model (2-4 weeks)

**Objective:** Measure predictive power, not correlation

**Tasks:**
1. Implement k-NN classifier (already prototyped)
2. Baseline: Random Forest, XGBoost
3. Evaluate accuracy, precision, recall per scale
4. Cross-validation (geographic stratification)

**Success metric:** Accuracy > 2× random chance

### Phase 3: Quality Control Tool (4-8 weeks)

**Objective:** Practical application for metadata curation

**Tasks:**
1. Outlier detection pipeline (GE vs ENVO disagreement)
2. Web interface for curator review
3. Automated suggestion system (k-NN predictions)
4. Integration with BioSample submission workflow

**Deliverable:** Live tool for NCBI/EBI metadata curation

### Phase 4: Publication (4-8 weeks)

**Objective:** Share findings with community

**Tasks:**
1. Write manuscript (methods, results, discussion)
2. Create supplementary materials (notebook, data, code)
3. Submit to bioinformatics journal (e.g., Bioinformatics, NAR)
4. Preprint on bioRxiv

**Impact:** Influence metadata standards and curation practices

---

## 🔬 Technical Considerations

### Statistical Power

**Current:**
- 246 samples
- 6,826 pairs (after filtering)
- Power to detect r > 0.3 at α = 0.05

**Needed for r = 0.2 detection:**
- ~800 samples
- ~320,000 pairs
- Better for weak correlation studies

### Computational Cost

**Current pipeline:**
- Google Earth API: ~2 seconds per sample
- ENVO embeddings: cached (instant for duplicates)
- Pairwise similarities: O(n²) for n samples

**Scaling to 10,000 samples:**
- Embeddings: ~5 hours (Google Earth API limited)
- Similarities: ~10 minutes (embarrassingly parallel)
- Storage: ~500 MB (embeddings + metadata)

**Bottleneck:** Google Earth Engine API rate limits

### Geographic Stratification

**For unbiased sampling:**
```python
# Stratify by major biomes
biomes = {
    'terrestrial': 0.4,  # 40% of samples
    'marine': 0.3,       # 30%
    'freshwater': 0.15,  # 15%
    'urban': 0.10,       # 10%
    'extreme': 0.05      # 5%
}

# Stratify by latitude
latitude_bins = {
    'arctic': 66-90°,
    'temperate_north': 23-66°,
    'tropical': -23-23°,
    'temperate_south': -66--23°,
    'antarctic': -90--66°
}
```

### Validation Strategy

**Cross-validation approaches:**
1. **K-fold:** Random split (ignores geography)
2. **Geographic CV:** Hold out regions (tests generalization)
3. **Biome CV:** Hold out biome types (tests diversity)
4. **Temporal CV:** Hold out years (tests temporal stability)

**Recommended:** Geographic CV to avoid spatial autocorrelation

---

## 📚 Related Work

### Similar Studies

**Satellite + Metadata:**
- iNaturalist species distribution models
- GBIF occurrence data validation
- Land cover classification from satellite

**Embedding Similarity:**
- Sentence embeddings for semantic similarity
- Image embeddings for reverse image search
- Cross-modal embedding alignment (vision + language)

### Gaps This Work Fills

1. **Environmental metadata:** First to compare satellite vs ontology embeddings
2. **Degenerate filtering:** Demonstrates methodological pitfall
3. **Negative result:** Published weak correlations are rare but valuable
4. **Practical tool:** k-NN prediction for metadata curation

---

## 🎓 Lessons Learned

### Methodological

1. **Always filter degenerate pairs** in similarity studies
2. **Spearman before Pearson** when relationship is unknown
3. **Report effect sizes** (r values), not just p-values
4. **Visualize before correlating** (scatter plots reveal non-linearity)

### Scientific

1. **Negative results are valuable** when rigorously tested
2. **Weak correlation ≠ no information** (still useful for classification)
3. **Complementary is better than redundant** (different perspectives matter)

### Practical

1. **Start small, scale deliberately** (246 → 1,000 → 10,000)
2. **Build infrastructure first** (caching, filtering, reproducibility)
3. **Document as you go** (notebook + markdown files)

---

## 🚀 Next Actions

### Immediate (This Week)

- [ ] Review findings with collaborators
- [ ] Decide on direction (scale up vs pivot vs publish)
- [ ] Plan next dataset (if scaling up)

### Short-term (This Month)

- [ ] Process 500-1,000 more samples (if scaling)
- [ ] Implement classification model (if pivoting)
- [ ] Draft introduction (if publishing)

### Long-term (Next Quarter)

- [ ] Complete analysis on larger dataset
- [ ] Build quality control prototype
- [ ] Submit manuscript

---

## 📝 Final Thoughts

This analysis demonstrates the value of **methodological rigor** and the importance of **negative results** in science. The weak correlation between satellite imagery and ENVO annotations is not a failure—it's a discovery that these two perspectives capture **complementary** aspects of environmental context.

**The satellite sees:** Physical landscape, land cover, terrain
**ENVO captures:** Biological context, ecosystem type, functional role

**Together, not separately, they provide comprehensive environmental metadata.**

The path forward depends on your goals:
- **Basic science:** Understand why correlation is weak (scale up, investigate)
- **Applied tool:** Build prediction/validation system (pivot to classification)
- **Community impact:** Publish findings, influence metadata standards

All three paths are valuable. The infrastructure you've built supports any direction.
