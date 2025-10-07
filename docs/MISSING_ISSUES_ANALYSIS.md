# Missing Issues Analysis: RF Enhancements

## Summary

After reviewing all issues, **NONE** of the following enhancements are tracked:

1. ❌ RF performance on other two fields (env_broad_scale, env_medium)
2. ❌ RF performance on combinations of fields (pairs or all three combined)
3. ❌ Partial success scoring using ontology hierarchy (parent/child relationships)

## What Exists vs What's Missing

### ✅ What IS Covered (Issue #49)

**Issue #49** ("Apply Random Forest ENVO predictor to NMDC complete dataset") addresses:
- Running RF on NMDC complete dataset (8,121 samples)
- Expanding from `env_local_scale` only to **all 3 MIxS scales**:
  - `env_broad_scale`
  - `env_local_scale`
  - `env_medium`
- Training **3 separate classifiers** (one per scale)
- Comparing RF to k-NN baseline

**Success criteria from #49:**
```
- [ ] Trains 3 classifiers: broad_scale, local_scale, medium
- [ ] Reports accuracy, precision, recall, F1 per scale
- [ ] Feature importance analysis for all 3 scales
- [ ] Performance comparison: RF vs k-NN
```

This covers **your question #1** ✅

### ❌ What is NOT Covered

#### Missing Issue #1: Combined/Joint Prediction

**Your question #2:** "How does it perform on combinations of the three fields?"

**Not addressed:** Training models that predict:
- **Pairs:**
  - broad_scale + local_scale jointly
  - broad_scale + medium jointly
  - local_scale + medium jointly
- **All three:** broad_scale + local_scale + medium as a single multi-output prediction

**Current approach (Issue #49):** 3 **independent** classifiers, each predicting one scale

**Missing approach:**
- **Multi-output RandomForest** - Predict all 3 scales simultaneously
- **Chained classifiers** - Use predicted broad_scale as feature for predicting local_scale
- **Joint encoding** - Treat triad as a single categorical variable (e.g., "ENVO:X|ENVO:Y|ENVO:Z")

**Research questions:**
- Does predicting all 3 together improve accuracy vs independent models?
- Are certain triads more predictable from satellite imagery?
- Can we use hierarchical constraints (e.g., if broad=marine, local can't be desert)?

---

#### Missing Issue #2: Ontology-Aware Evaluation

**Your question #3:** "Can we consider predictions of the parent or a child of the asserted value a 'partial success'?"

**Not addressed:**
- Using ENVO ontology hierarchy in evaluation
- Partial credit scoring
- Semantic similarity metrics

**Current evaluation (Issue #49):** Binary exact match
- Predicted `ENVO:00000447` vs True `ENVO:00000447` → ✅ Correct (1.0)
- Predicted `ENVO:00000448` vs True `ENVO:00000447` → ❌ Wrong (0.0)

**Missing evaluation approaches:**

##### 2a. Hierarchy-based Partial Credit

Use ENVO ontology structure to assign partial scores:

```python
def ontology_aware_score(predicted, true, ontology):
    """
    Score based on ontology distance.

    Examples:
        - Exact match: 1.0
        - Parent/child (1 hop): 0.75
        - Grandparent/grandchild (2 hops): 0.5
        - Sibling (same parent): 0.5
        - Distant relative: 0.25
        - Unrelated: 0.0
    """
    if predicted == true:
        return 1.0

    distance = ontology.shortest_path(predicted, true)
    relationship = ontology.get_relationship(predicted, true)

    if relationship == "parent" or relationship == "child":
        return 0.75
    elif relationship == "sibling":
        return 0.5
    elif distance <= 3:
        return 0.25
    else:
        return 0.0
```

**Example scenarios:**

```
True: ENVO:00000447 (marine biome)
  ├─ Predicted ENVO:00000447 (marine biome) → 1.0 ✅ exact
  ├─ Predicted ENVO:01000048 (coastal sea water) → 0.75 ⚠️ child term
  ├─ Predicted ENVO:00000428 (biome) → 0.75 ⚠️ parent term
  ├─ Predicted ENVO:01000174 (terrestrial biome) → 0.5 ⚠️ sibling
  └─ Predicted ENVO:00002011 (soil) → 0.0 ❌ unrelated
```

##### 2b. Semantic Similarity

Use ENVO embedding cosine similarity:

```python
def semantic_similarity_score(predicted, true, envo_embeddings):
    """
    Score based on embedding similarity between predicted and true terms.

    Examples:
        - Cosine similarity > 0.9: High similarity (partial success)
        - Cosine similarity 0.7-0.9: Moderate similarity
        - Cosine similarity < 0.7: Low similarity (failure)
    """
    pred_emb = envo_embeddings[predicted]
    true_emb = envo_embeddings[true]

    similarity = cosine_similarity(pred_emb, true_emb)

    # Convert similarity to score
    if similarity > 0.9:
        return 0.9
    elif similarity > 0.7:
        return 0.6
    else:
        return 0.0
```

##### 2c. Taxonomic Level Scoring

Different penalties for errors at different granularity levels:

```python
# For env_broad_scale (coarse) - stricter evaluation
#   Wrong biome = major error

# For env_local_scale (fine) - more lenient
#   Wrong local term but correct parent = minor error

# Weighted average:
#   broad_scale_score * 0.4 +
#   medium_score * 0.3 +
#   local_scale_score * 0.3
```

---

## Proposed New Issues

### Issue: Multi-Output Random Forest for Joint Triad Prediction

**Title:** Train multi-output RF to predict environmental triad jointly (broad + local + medium)

**Description:**
Currently (Issue #49) we train 3 independent Random Forest classifiers. Test whether predicting all 3 MIxS scales **simultaneously** improves performance.

**Approaches:**
1. **Multi-output RandomForestClassifier** - sklearn native support
2. **Chained classifiers** - Use broad_scale prediction as feature for medium/local
3. **Joint encoding** - Treat triad as single 553³ categorical space

**Research questions:**
- Does joint prediction improve accuracy?
- Which triads are most predictable from GE embeddings?
- Can hierarchical constraints help (e.g., marine broad → no desert local)?

**Comparison metrics:**
- Independent RF (3 models) vs Multi-output RF (1 model)
- Accuracy per scale
- Triad-level accuracy (all 3 correct)
- Execution time
- Feature importance differences

**Deliverables:**
- `notebooks/random_forest_joint_triad_prediction.ipynb`
- Comparison table: independent vs joint
- Visualization: triad confusion matrix

**Related:** #49 (independent classifiers), #31 (original classifier idea)

---

### Issue: Ontology-Aware Evaluation Metrics for ENVO Predictions

**Title:** Implement partial credit scoring using ENVO hierarchy (parent/child relationships)

**Description:**
Current RF evaluation uses binary exact match (correct/incorrect). This penalizes semantically close predictions equally with completely wrong predictions.

**Example problem:**
```
True: ENVO:00000447 (marine biome)
Predicted: ENVO:00000428 (biome) [parent term]
Current score: 0.0 (wrong)
Better score: 0.75 (close - parent term)
```

**Implementation tasks:**

1. **ENVO Ontology Integration**
   - Use `pronto` or `owlready2` to load ENVO.owl
   - Build graph structure for relationship queries
   - Cache ontology in repo or download on-demand

2. **Hierarchy-Based Scoring**
   - Exact match: 1.0
   - Parent/child (1 hop): 0.75
   - Grandparent/grandchild or sibling: 0.5
   - Distant (3+ hops): 0.25
   - Unrelated: 0.0

3. **Semantic Similarity Scoring**
   - Use ENVO embeddings (already have 1536-dim vectors)
   - Cosine similarity between predicted and true term
   - Scale to 0-1 score

4. **New Metrics**
   - `hierarchical_accuracy`: Mean hierarchy-aware score
   - `semantic_accuracy`: Mean embedding-similarity score
   - `top_k_hierarchical`: True term in top-k by hierarchy distance
   - Per-scale analysis: Which scale benefits most from partial credit?

5. **Visualization**
   - Heatmap: True vs Predicted with hierarchy coloring
   - Scatter: Exact accuracy vs hierarchical accuracy per class
   - Examples: "Close misses" that get partial credit

**Use cases:**
- More nuanced evaluation of RF performance
- Identify when model is "close" vs "completely wrong"
- Inform active learning (focus on truly confused samples, not close calls)
- Better comparison to k-NN (which may predict related terms)

**Deliverables:**
- `src/env_embeddings/ontology_metrics.py` - Scoring functions
- Updated RF notebook with hierarchy-aware metrics
- Comparison: exact vs hierarchical accuracy
- Analysis: Which ENVO terms have closest "near misses"?

**Related:** #49 (RF training), #31 (original classifier), OLS integration in codebase

---

### Issue: Analyze RF Feature Importance Across Triad Components

**Title:** Compare which GE embedding dimensions predict broad vs local vs medium scales

**Description:**
Issue #49 trains 3 independent RF classifiers (broad, local, medium). Analyze whether different satellite features predict different ENVO scales.

**Research questions:**
- Do different GE embedding dimensions matter for different scales?
- Is broad_scale predicted by coarse features (low-freq bands)?
- Is local_scale predicted by fine features (high-freq bands)?
- Can we prune features per-scale for efficiency?

**Analysis tasks:**

1. **Feature Importance Comparison**
   - Plot top 20 features for each scale side-by-side
   - Identify unique vs shared important dimensions
   - Correlation: broad_importance vs local_importance

2. **Dimension Specialization**
   - Which dimensions predict only broad? only local? only medium?
   - Which are universal (important for all 3)?

3. **Earth Engine Band Mapping**
   - Map GE_dim_0 through GE_dim_63 to actual EE bands
   - Identify: Are blue bands more important than IR? NDVI vs NDWI?
   - Physical interpretation of top features

4. **Efficiency Analysis**
   - Can we train with fewer features per scale?
   - Trade-off: accuracy vs speed

**Deliverables:**
- Feature importance comparison plots
- Venn diagram: shared vs unique important features
- Table: Top features per scale with EE band interpretation
- Recommendation: Keep all 64 dims or specialize per scale?

**Related:** #49 (trains 3 classifiers), EE embedding code

---

## Priority Ranking

### High Priority
1. **Ontology-Aware Evaluation** ⭐⭐⭐
   - Most scientifically valuable
   - Relatively easy to implement (ENVO already integrated)
   - Improves evaluation quality immediately
   - No retraining needed - just scoring changes

### Medium Priority
2. **Multi-Output RF** ⭐⭐
   - Interesting research question
   - Moderate implementation effort
   - May or may not improve accuracy (experiment needed)
   - Requires retraining models

3. **Feature Importance Analysis** ⭐⭐
   - Good for interpretation
   - Easy to add to existing #49 work
   - Helps understand what satellite sees
   - No additional training needed

### Lower Priority
4. **Pair-wise Prediction** ⭐
   - Less clear value over independent or joint
   - More complex to evaluate (which pairs matter?)
   - Could be subset of multi-output work

---

## Recommended Next Steps

### Option 1: Add to Issue #49 (Current PR/Branch)
Expand #49 success criteria to include:
- [x] Train 3 independent classifiers
- [ ] Implement ontology-aware scoring
- [ ] Compare exact vs hierarchical accuracy
- [ ] Analyze feature importance across scales

**Pros:** Keep work consolidated, faster iteration
**Cons:** Makes #49 larger in scope

### Option 2: Create Separate Issues (Better)
Keep #49 focused on basic RF training, create new issues for:
- Issue #50: Ontology-aware evaluation (post #49)
- Issue #51: Multi-output joint triad prediction (post #49)
- Issue #52: Feature importance comparison (during/after #49)

**Pros:** Clear separation of concerns, parallel work possible
**Cons:** More issues to track

### Option 3: Milestone-Based
Create milestone "RF Enhancements" with sub-issues:
- Milestone: Advanced RF Analysis
  - #49: Basic 3-classifier training ← current
  - #50: Ontology evaluation
  - #51: Joint prediction
  - #52: Feature analysis

**Pros:** Shows progression, organized roadmap
**Cons:** Requires milestone management

---

## Technical Implementation Notes

### Ontology-Aware Scoring

**Dependencies needed:**
```python
# Add to pyproject.toml
pronto = "^2.5.0"  # OBO/OWL parsing
networkx = "^3.0"  # Graph operations (already have?)
```

**Code structure:**
```python
# src/env_embeddings/ontology_metrics.py
from pronto import Ontology
import networkx as nx

class ENVOHierarchy:
    def __init__(self, owl_path="data/envo.owl"):
        self.ontology = Ontology(owl_path)
        self.graph = self._build_graph()

    def distance(self, term1, term2):
        """Shortest path in ontology graph"""
        return nx.shortest_path_length(self.graph, term1, term2)

    def hierarchical_score(self, predicted, true):
        """Score based on ontology distance"""
        if predicted == true:
            return 1.0

        try:
            dist = self.distance(predicted, true)
            if dist == 1:
                return 0.75  # parent/child
            elif dist == 2:
                return 0.5   # grandparent or sibling
            elif dist <= 3:
                return 0.25
            else:
                return 0.0
        except nx.NetworkXNoPath:
            return 0.0  # unrelated branches

# Usage in notebook:
envo_hierarchy = ENVOHierarchy()
scores = [envo_hierarchy.hierarchical_score(pred, true)
          for pred, true in zip(y_pred, y_test)]
hierarchical_accuracy = np.mean(scores)
```

### Multi-Output RF

**sklearn native support:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Create targets matrix (n_samples, 3)
y_multi = df[['env_broad_scale', 'env_local_scale', 'env_medium']].values

# Train multi-output model
rf_multi = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=100, max_depth=10)
)
rf_multi.fit(X_train, y_multi_train)

# Predict all 3 scales at once
y_pred_multi = rf_multi.predict(X_test)  # shape: (n_test, 3)

# Evaluate per-scale and triad-level
broad_acc = accuracy_score(y_multi_test[:, 0], y_pred_multi[:, 0])
local_acc = accuracy_score(y_multi_test[:, 1], y_pred_multi[:, 1])
medium_acc = accuracy_score(y_multi_test[:, 2], y_pred_multi[:, 2])

# Triad-level accuracy (all 3 must match)
triad_correct = (y_multi_test == y_pred_multi).all(axis=1)
triad_acc = triad_correct.mean()
```

### Feature Importance Comparison

**Simple implementation in existing notebook:**
```python
# After training 3 classifiers in Issue #49

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (scale, clf) in enumerate([
    ('broad_scale', rf_broad),
    ('local_scale', rf_local),
    ('medium', rf_medium)
]):
    importances = clf.feature_importances_
    top_20 = np.argsort(importances)[-20:]

    axes[idx].barh(range(20), importances[top_20])
    axes[idx].set_title(f'Top 20 Features: {scale}')
    axes[idx].set_xlabel('Importance')
    axes[idx].set_ylabel('GE Dimension')

plt.tight_layout()
plt.show()

# Correlation analysis
broad_imp = rf_broad.feature_importances_
local_imp = rf_local.feature_importances_
medium_imp = rf_medium.feature_importances_

print(f"Broad-Local correlation: {np.corrcoef(broad_imp, local_imp)[0,1]:.3f}")
print(f"Broad-Medium correlation: {np.corrcoef(broad_imp, medium_imp)[0,1]:.3f}")
print(f"Local-Medium correlation: {np.corrcoef(local_imp, medium_imp)[0,1]:.3f}")
```
