# GitHub Issues Created - Random Forest Enhancement Roadmap

## Summary

Created **7 new issues** to address missing functionality and important considerations for Random Forest ENVO prediction.

## Issues Created

### Priority: HIGH

#### Issue #54: Handle duplicate coordinates with same/different ENVO annotations
**URL**: https://github.com/contextualizer-ai/env-embeddings/issues/54

**Problem**: Multiple samples with identical coordinates/dates → identical GE embeddings but potentially different ENVO annotations

**Impact**:
- Creates impossible learning task (X → Y₁ AND X → Y₂)
- Inflates dataset via pseudo-replication
- Test set leakage if duplicates split across train/test

**Solutions proposed**:
1. Deduplicate by coordinate (conservative)
2. Majority vote for conflicts
3. Weight duplicates lower
4. Flag and exclude conflicts only

**Recommendation**: Start with deduplication for clean baseline (Option 1)

**Related to**: #49 (needs this before RF training)

---

#### Issue #51: Implement ontology-aware evaluation metrics (partial credit)
**URL**: https://github.com/contextualizer-ai/env-embeddings/issues/51

**Problem**: Binary exact match doesn't give credit for "close" predictions (e.g., predicting parent term)

**Example**:
```
True: ENVO:00000447 (marine biome)
Predicted: ENVO:00000428 (biome) [parent term]
Current score: 0.0
Better score: 0.75 (hierarchically close)
```

**Implementation**:
- Use ENVO ontology graph (via `pronto` library)
- Hierarchy-based scoring: exact=1.0, parent/child=0.75, sibling=0.5, distant=0.25
- Semantic similarity scoring: Use ENVO embeddings cosine similarity
- New metrics: hierarchical_accuracy, semantic_accuracy, top_k_hierarchical

**Impact**: More nuanced evaluation, distinguishes "close misses" from "completely wrong"

**Related to**: #49 (improves evaluation), #31 (original classifier idea)

---

### Priority: MEDIUM-HIGH

#### Issue #56: Compare RF predictions to k-NN baseline
**URL**: https://github.com/contextualizer-ai/env-embeddings/issues/56

**Question**: Does supervised learning (RF) beat unsupervised baseline (k-NN)?

**k-NN baseline** (already exists in similarity_analysis.ipynb):
- Find k=5 most GE-similar samples
- Majority vote their ENVO terms
- Simple, interpretable, no training

**Comparison metrics**:
- Accuracy (exact match)
- Hierarchical accuracy (from #51)
- Prediction confidence
- Runtime (k-NN slow at inference)

**Expected outcome**:
- If RF >> k-NN: ML justified
- If RF ≈ k-NN: Consider simpler k-NN approach
- If k-NN > RF: GE embeddings designed for similarity, not classification

**Related to**: #49 (RF baseline), #51 (evaluation), similarity_analysis.ipynb

---

### Priority: MEDIUM

#### Issue #52: Multi-output Random Forest for joint triad prediction
**URL**: https://github.com/contextualizer-ai/env-embeddings/issues/52

**Current approach** (#49): 3 independent classifiers (broad, local, medium)

**Proposed**: Predict all 3 scales jointly to exploit dependencies

**Approaches**:
1. **Multi-output RF**: sklearn native, predicts all 3 simultaneously
2. **Chained classifiers**: Use predicted broad as feature for local/medium
3. **Joint encoding**: Treat triad as single categorical (huge class space)

**Research questions**:
- Does joint prediction improve accuracy?
- Can we use hierarchical constraints? (e.g., marine broad → no desert local)
- Which triads are most/least predictable?

**Evaluation**:
- Per-scale accuracy vs independent (#49 baseline)
- Triad-level accuracy (all 3 must match)
- Constraint violation analysis

**Related to**: #49 (provides baseline for comparison)

---

#### Issue #53: Analyze GE feature importance across triad scales
**URL**: https://github.com/contextualizer-ai/env-embeddings/issues/53

**Question**: Do different GE embedding dimensions predict different ENVO scales?

**Analysis**:
1. Compare `feature_importances_` across 3 RF classifiers
2. Identify unique vs shared important features (Venn diagram)
3. Map GE dimensions to Earth Engine bands (blue, green, NIR, NDVI, etc.)
4. Test reduced feature sets (efficiency vs accuracy trade-off)

**Hypotheses**:
- Broad scale: Coarse features (band means, overall NDVI)
- Local scale: Fine features (texture, variance, edges)
- Some universal features: NDVI for vegetation

**Deliverables**:
- Side-by-side feature importance plots
- Correlation analysis between scales
- Physical interpretation (which bands matter?)
- Efficiency recommendation (64 dims or specialize?)

**Related to**: #49 (provides feature_importances_), earth_engine.py

---

#### Issue #55: Scale RF training to NCBI complete (382,955 samples)
**URL**: https://github.com/contextualizer-ai/env-embeddings/issues/55

**Goal**: Test scalability on NCBI complete - **47x larger** than NMDC

**Challenges**:
1. **Execution time**: Estimated 45-90 minutes (vs 2-3 min for NMDC)
2. **Class imbalance**: More diverse ENVO terms, many with 1-2 samples
3. **Data quality**: Uncurated, more noise than NMDC
4. **Duplicates**: Higher rate expected (Issue #54)

**Optimizations**:
- Reduce trees: 100 → 50
- Subsample per tree: `max_samples=0.5`
- Skip cross-validation initially
- Progress monitoring: `verbose=2`

**Cross-dataset evaluation**:
- Train on NCBI, test on NMDC
- Train on NMDC, test on NCBI
- Does NMDC curation improve model quality?

**Expected results**:
- NCBI accuracy: 45-50% (vs NMDC 50-55%)
- If NCBI << NMDC: Too noisy for supervised learning
- Runtime: 20-30 minutes (optimized)

**Related to**: #49 (NMDC baseline), #54 (duplicate handling), #46 (NCBI dataset)

---

#### Issue #57: Document ENVO term coverage and class distribution
**URL**: https://github.com/contextualizer-ai/env-embeddings/issues/57

**Goal**: Understand what we can/can't predict based on data coverage

**Analysis**:
1. **Per-dataset statistics**: Unique terms, class distribution, Gini coefficient
2. **Cross-dataset comparison**: Venn diagram of term overlap
3. **ENVO ontology coverage**: What % of 6000+ ENVO terms do we have?
4. **Predictability analysis**: Does sample count correlate with RF accuracy?

**Expected findings**:
- NMDC: ~200-400 unique terms per scale, ~5-10% ENVO coverage
- NCBI: ~800-1500 unique terms, ~15-25% coverage, heavily skewed (Zipf's law)
- GOLD: ~300-500 terms

**Implications**:
- Need `class_weight='balanced'` for skewed distributions
- Filter rare classes (< 5 samples)
- Model only works for covered ENVO subset (closed-world)

**Deliverables**:
- `notebooks/envo_coverage_analysis.ipynb`
- Class distribution visualizations (log scale histograms, Zipf plots)
- Ontology tree showing covered branches
- Recommendations for data collection

**Related to**: #40 (original question), #49 (RF training), #55 (NCBI)

---

## Dependency Graph

```
Priority order and dependencies:

HIGH (Do first):
  #54 Duplicate handling ─────┐
                              ├──> #49 RF training (existing, update)
  #51 Ontology-aware eval ────┘

MEDIUM-HIGH (After #49):
  #56 k-NN comparison ────────> Needs #49 baseline

MEDIUM (Parallel or after #49):
  #53 Feature importance ─────> Can add to #49 or separate
  #57 ENVO coverage ──────────> Parallel to #49
  #52 Multi-output RF ────────> After #49 baseline

MEDIUM (Later):
  #55 NCBI scaling ───────────> After #49, #54, #57
```

## Implementation Roadmap

### Phase 1: Clean Foundation (Current PR #50)
- [x] Merge RF notebook with NMDC dataset (#49)
- [x] Update dataset path
- [ ] Add duplicate handling (#54)
- [ ] Expand to all 3 scales (#49)

### Phase 2: Enhanced Evaluation
- [ ] Implement ontology-aware scoring (#51)
- [ ] Add k-NN comparison (#56)
- [ ] Feature importance analysis (#53)

### Phase 3: Advanced Methods
- [ ] ENVO coverage analysis (#57)
- [ ] Multi-output RF (#52)

### Phase 4: Production Scale
- [ ] NCBI complete training (#55)
- [ ] Cross-dataset validation

## Summary Statistics

**Total issues created**: 7
- **High priority**: 2 (#54, #51)
- **Medium-high priority**: 1 (#56)
- **Medium priority**: 4 (#52, #53, #55, #57)

**Estimated implementation time**:
- #54: 2-4 hours (add to #49)
- #51: 1-2 days (new module + integration)
- #56: 4-8 hours (extract k-NN, compare)
- #53: 2-4 hours (add to #49 notebook)
- #57: 1-2 days (separate analysis notebook)
- #52: 2-3 days (new approaches, evaluation)
- #55: 2-3 days (optimization, cross-validation)

**Total**: ~2-3 weeks of focused work (if done sequentially)

## Open Questions for Discussion

1. **Issue #52 (Multi-output)**: Worth the complexity? Or stick with independent classifiers?
2. **Issue #54 (Duplicates)**: Should we exclude conflicts entirely or try to learn from them?
3. **Issue #55 (NCBI)**: Is NCBI too noisy? Need curation step first?
4. **Issue #51 (Ontology)**: What hierarchy distance scoring makes sense? (1 hop=0.75, 2 hops=0.5?)
5. **Issue #56 (k-NN)**: If k-NN performs well, is RF worth the complexity?

## Next Steps

**Immediate** (add to current PR #50):
1. Address #54 (duplicate handling) in RF notebook
2. Ensure #49 success criteria complete (3 classifiers trained)

**Short-term** (next 1-2 weeks):
3. Implement #51 (ontology metrics) - high scientific value
4. Add #56 (k-NN comparison) - validates ML approach
5. Add #53 (feature importance) - easy addition to #49

**Medium-term** (next month):
6. Create #57 (ENVO coverage analysis)
7. Experiment with #52 (multi-output RF)
8. Scale to #55 (NCBI complete)
