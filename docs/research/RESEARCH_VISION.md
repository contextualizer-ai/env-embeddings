# Research Vision: Environmental Context Quality Control and Prediction

## The Big Picture

You're building a **quality control and recommendation system** for environmental metadata using the divergence between satellite imagery (objective, physical) and ontological annotations (subjective, curator-provided).

## Core Insight

**Disagreement between Google Earth embeddings and ENVO embeddings is informative:**

- **High GE similarity + High ENVO similarity** → Good metadata, samples are genuinely similar
- **High GE similarity + Low ENVO similarity** → **Metadata quality issue** - physically similar locations have inconsistent annotations
- **Low GE similarity + High ENVO similarity** → Different locations can share environmental context (e.g., both "forest" but different continents)

## Three-Phase Research Pipeline

### Phase 1: Find Optimal ENVO Combination ✅ (Current Work)

**Goal:** Determine which combination of MIxS environmental context variables best predicts satellite-based similarity

**Approaches:**
1. Individual scales: `broad_scale`, `medium`, `local_scale`
2. Concatenation: `[broad; medium; local]` (4608-dim)
3. Weighted average: `(broad + medium + local) / 3` (1536-dim)
4. Learned weights: `w1·broad + w2·medium + w3·local` (future)

**Success Metric:** Highest Pearson correlation with GE embeddings after removing degenerate pairs

**Current Status:** Implemented in notebook, will reveal best combination

---

### Phase 2: Identify and Analyze Outliers 🎯 (Next Step)

**Goal:** Find samples with suspicious metadata by detecting divergence between physical and semantic similarity

#### 2A. High GE / Low ENVO Outliers (Metadata Quality Issues)

**Example Scenario:**
```
Sample A: (35.7°N, 139.8°E, 2018) - ENVO: "terrestrial biome | soil | agricultural field"
Sample B: (35.7°N, 139.8°E, 2018) - ENVO: "marine biome | sea water | coastal zone"

GE Similarity: 0.99 (same location, same date → nearly identical satellite view)
ENVO Similarity: 0.25 (completely different biomes!)

→ FLAG: One of these annotations is wrong
```

**Detection Method:**
```python
# Define outliers as pairs in the upper-right quadrant
outliers_metadata_issue = pairs_filtered[
    (pairs_filtered['ge_similarity'] > 0.8) &  # Physically very similar
    (pairs_filtered['envo_best_similarity'] < 0.3)  # Semantically very different
]
```

**Analysis Questions:**
1. How many outlier pairs exist?
2. Do certain biosamples appear repeatedly in outlier pairs? (chronic bad metadata)
3. Are there patterns? (e.g., marine/terrestrial confusion, missing terms)

#### 2B. Low GE / High ENVO Outliers (Interesting Biology)

**Example Scenario:**
```
Sample A: (34.0°N, 118.2°W) - California desert
Sample B: (31.2°N, 29.9°E) - Egyptian desert

GE Similarity: 0.15 (different continents, different geology)
ENVO Similarity: 0.95 (both "desert biome | sand | arid soil")

→ VALID: Similar environment types in different locations
```

These are **not errors** - they show that ENVO terms successfully capture environmental similarity across geography.

---

### Phase 3: Metadata Prediction and Improvement 🚀 (Future Work)

**Goal:** Use satellite imagery to propose or validate ENVO triads

#### 3A. Predict Missing Triads

**Scenario:** Sample has coordinates/date but missing ENVO terms

**Method:**
```python
def predict_envo_triad(sample_coords, sample_date, training_data):
    """
    Predict ENVO triad for a sample with missing metadata.

    1. Get Google Earth embedding for target sample
    2. Find k-nearest neighbors by GE embedding similarity
    3. Aggregate their ENVO triads (majority vote or weighted average)
    4. Return predicted broad/medium/local scale terms
    """

    # Get GE embedding for target
    target_ge_embedding = get_embedding(lat, lon, year)

    # Find similar samples by satellite imagery
    similarities = []
    for train_sample in training_data:
        sim = cosine_similarity(target_ge_embedding, train_sample.ge_embedding)
        similarities.append((sim, train_sample))

    # Get top-k most similar samples
    top_k = sorted(similarities, reverse=True)[:10]

    # Aggregate their ENVO terms (majority vote)
    broad_candidates = [s.envo_broad for _, s in top_k]
    medium_candidates = [s.envo_medium for _, s in top_k]
    local_candidates = [s.envo_local for _, s in top_k]

    return {
        'broad_scale': most_common(broad_candidates),
        'medium': most_common(medium_candidates),
        'local_scale': most_common(local_candidates),
        'confidence': top_k[0][0]  # similarity to nearest neighbor
    }
```

**Validation:**
- Hold out 20% of samples with known triads
- Predict their triads using only GE embeddings
- Measure accuracy: How often do we get the exact triad correct?

#### 3B. Flag Suspect Triads

**Scenario:** Sample has ENVO triad but it disagrees with satellite imagery

**Method:**
```python
def flag_suspect_triads(sample, training_data, threshold=0.7):
    """
    Flag samples whose ENVO triad is inconsistent with their satellite imagery.

    Returns:
        List of alternative triad suggestions with confidence scores
    """

    # Find samples with similar GE embeddings (physically similar)
    similar_by_satellite = find_similar_samples(
        sample.ge_embedding,
        training_data,
        similarity_threshold=0.8
    )

    # Check if their ENVO triads agree
    triad_agreement = []
    for sim_sample in similar_by_satellite:
        envo_sim = cosine_similarity(
            sample.envo_embedding,
            sim_sample.envo_embedding
        )
        triad_agreement.append(envo_sim)

    avg_agreement = mean(triad_agreement)

    if avg_agreement < threshold:
        # This sample's triad disagrees with physically similar samples
        # Propose alternative triads from the similar samples
        alternative_triads = aggregate_triads(similar_by_satellite)

        return {
            'flagged': True,
            'reason': f'Low ENVO agreement ({avg_agreement:.2f}) with physically similar samples',
            'current_triad': sample.envo_triad,
            'suggested_alternatives': alternative_triads,
            'similar_samples': [s.accession for s in similar_by_satellite[:5]]
        }

    return {'flagged': False}
```

**Use Cases:**
1. **Quality Control:** Curators review flagged samples before publication
2. **Batch Correction:** Identify systematic errors (e.g., swapped broad/local scales)
3. **Metadata Enrichment:** Suggest triads for older samples lacking structured metadata

#### 3C. Active Learning for Metadata Improvement

**Goal:** Prioritize which samples to manually review

**Method:**
```python
def prioritize_for_manual_review(samples):
    """
    Rank samples by potential metadata impact.

    Priority factors:
    1. High GE/ENVO disagreement (likely error)
    2. Many other samples depend on this one (high influence)
    3. Rare environmental context (filling gaps in training data)
    """

    scores = []
    for sample in samples:
        # Factor 1: Disagreement score
        disagreement = calculate_ge_envo_disagreement(sample)

        # Factor 2: Influence score (how many samples are similar to this one?)
        influence = count_similar_samples(sample, threshold=0.8)

        # Factor 3: Rarity score (is this an underrepresented environment?)
        rarity = calculate_envo_rarity(sample.envo_triad)

        # Combined priority score
        priority = 0.5 * disagreement + 0.3 * influence + 0.2 * rarity
        scores.append((priority, sample))

    return sorted(scores, reverse=True)
```

---

## Concrete Next Steps

### Immediate (This Week)

1. **Run updated notebook** to determine best ENVO combination
2. **Add outlier detection cell** to identify high-GE/low-ENVO pairs
3. **Analyze outlier patterns:**
   - Which samples appear most frequently?
   - Are there systematic errors (e.g., marine/terrestrial swaps)?
   - Manual inspection of top 10 outlier pairs

### Short-term (Next Sprint)

4. **Build triad prediction prototype:**
   - K-nearest neighbors by GE embedding
   - Majority vote for ENVO terms
   - Cross-validation to measure accuracy

5. **Create suspect triad report:**
   - List samples with low ENVO agreement among GE-similar samples
   - Generate suggested alternative triads
   - Export for manual review

### Medium-term (Research Paper)

6. **Systematic evaluation:**
   - Holdout test set (20% of samples)
   - Measure triad prediction accuracy
   - Compare to baseline (random, most-frequent, etc.)

7. **Case studies:**
   - Manual review of flagged samples
   - Work with domain experts to validate predictions
   - Quantify metadata improvement

8. **Scale up:**
   - Process full BioSample database (not just 246 samples)
   - Build web interface for curators
   - Integrate with submission pipelines

---

## Example Analysis: Outlier Investigation

**Notebook cell to add:**

```python
print("=== OUTLIER ANALYSIS: METADATA QUALITY ISSUES ===")

# Define outliers: High GE similarity but Low ENVO similarity
outliers = pairs_filtered[
    (pairs_filtered['ge_similarity'] > 0.8) &
    (pairs_filtered['envo_best_similarity'] < 0.3)  # Use best ENVO combination
]

print(f"\nFound {len(outliers)} outlier pairs (high GE, low ENVO)")
print(f"  {len(outliers)/len(pairs_filtered)*100:.1f}% of all pairs\n")

# Identify samples that appear frequently in outlier pairs
from collections import Counter

outlier_samples = []
for _, pair in outliers.iterrows():
    outlier_samples.append(pair['accession_1'])
    outlier_samples.append(pair['accession_2'])

sample_counts = Counter(outlier_samples)
chronic_outliers = sample_counts.most_common(10)

print("Samples appearing most frequently in outlier pairs:")
print("(These may have incorrect metadata)\n")
for accession, count in chronic_outliers:
    sample = df_clean[df_clean['accession'] == accession].iloc[0]
    print(f"{accession}: {count} outlier pairs")
    print(f"  Location: ({sample['latitude']}, {sample['longitude']})")
    print(f"  ENVO: {sample['env_broad_scale']} | {sample['env_medium']} | {sample['env_local_scale']}")
    print()

# Show example outlier pairs
print("\n=== EXAMPLE OUTLIER PAIRS (for manual inspection) ===")
for idx, (_, pair) in enumerate(outliers.head(5).iterrows()):
    print(f"\nOutlier {idx+1}:")
    print(f"  GE Similarity: {pair['ge_similarity']:.3f}")
    print(f"  ENVO Similarity: {pair['envo_best_similarity']:.3f}")
    print(f"  Δ (disagreement): {pair['ge_similarity'] - pair['envo_best_similarity']:.3f}")

    sample1 = df_clean[df_clean['accession'] == pair['accession_1']].iloc[0]
    sample2 = df_clean[df_clean['accession'] == pair['accession_2']].iloc[0]

    print(f"\n  Sample 1: {sample1['accession']}")
    print(f"    Location: ({sample1['latitude']:.4f}, {sample1['longitude']:.4f})")
    print(f"    ENVO: {sample1['env_broad_scale']} | {sample1['env_medium']} | {sample1['env_local_scale']}")

    print(f"\n  Sample 2: {sample2['accession']}")
    print(f"    Location: ({sample2['latitude']:.4f}, {sample2['longitude']:.4f})")
    print(f"    ENVO: {sample2['env_broad_scale']} | {sample2['env_medium']} | {sample2['env_local_scale']}")

    print(f"\n  → DIAGNOSIS: Samples are physically very similar (GE={pair['ge_similarity']:.2f})")
    print(f"               but have very different ENVO terms (ENVO={pair['envo_best_similarity']:.2f})")
    print(f"               One of these metadata annotations is likely incorrect.")
```

---

## Expected Outcomes

### Scientific Contributions

1. **Quantify metadata quality:** What % of BioSample metadata is inconsistent with satellite imagery?

2. **Automate quality control:** Flag suspect samples before publication

3. **Metadata prediction:** Provide triads for samples lacking structured annotations

4. **Training data curation:** Identify high-quality samples for ML training

### Practical Impact

1. **Reduce curator burden:** Automated suggestions reduce manual annotation time

2. **Improve data quality:** Catch errors early in submission pipeline

3. **Enable better science:** Higher quality metadata → better meta-analyses

4. **Fill metadata gaps:** Millions of samples lack ENVO triads - we can predict them

---

## Technical Considerations

### Challenges

1. **ENVO term granularity:**
   - "forest" vs "temperate forest" vs "temperate coniferous forest"
   - Should we compare at different ontology depths?

2. **Temporal changes:**
   - Satellite imagery from 2018, sample collected in 2010
   - Land use changes (urban development, deforestation)

3. **Scale mismatch:**
   - GE embeddings: 10m resolution
   - ENVO broad_scale: biome-level (100s of km)
   - Which scale should agree?

4. **Training data bias:**
   - Current 246 samples may not represent all environments
   - Need larger, more diverse training set

### Solutions

1. **Hierarchical ENVO comparison:**
   - Compare at multiple ontology depths
   - Weight by specificity (more specific = higher weight)

2. **Temporal filtering:**
   - Only use samples where collection_date ≈ satellite_date
   - Or model temporal changes explicitly

3. **Multi-scale analysis:**
   - Compare local_scale to local GE features
   - Compare broad_scale to regional GE patterns

4. **Active learning:**
   - Iteratively expand training set
   - Prioritize diverse environments

---

## Summary: Your Vision

You're building a **virtuous cycle** for environmental metadata:

```
┌─────────────────────────────────────────────────────┐
│  Satellite Imagery (Objective, Physical)            │
│  ↓                                                   │
│  Google Earth Embeddings                            │
│  ↓                                                   │
│  Compare with ENVO Embeddings (Semantic, Curated)   │
│  ↓                                                   │
│  Identify Outliers (High GE / Low ENVO)             │
│  ↓                                                   │
│  Flag Suspect Metadata                              │
│  ↓                                                   │
│  Predict/Suggest Corrections                        │
│  ↓                                                   │
│  Human Review & Correction                          │
│  ↓                                                   │
│  Improved Training Data                             │
│  ↓                                                   │
│  Better Predictions (loop back to top)              │
└─────────────────────────────────────────────────────┘
```

This is **quality control through cross-validation** between independent data sources (satellite vs curator), using embedding similarity as the bridge.

Brilliant research direction! 🚀
