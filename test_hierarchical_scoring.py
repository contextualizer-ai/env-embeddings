#!/usr/bin/env python3
"""Quick test of hierarchical scoring on NMDC data.

Run with: uv run test_hierarchical_scoring.py

This loads NMDC data, trains a small RF model, and computes hierarchical scores
to validate the implementation before committing.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.env_embeddings.rf_analysis import load_source_data, train_rf_model
from src.env_embeddings.ontology_metrics import (
    ENVOHierarchy,
    hierarchical_score,
)

RANDOM_STATE = 42
NMDC_FILE = Path(
    "data/nmdc_flattened_biosample_for_env_embeddings_202510061052_complete.csv"
)

print("\n" + "=" * 80)
print("HIERARCHICAL SCORING VALIDATION TEST")
print("=" * 80)

# 1. Load NMDC data
print("\n1. Loading NMDC data (NMDC only - cleanest source)...")
df = load_source_data(NMDC_FILE, "NMDC", deduplicate=False)

if df is None:
    print("✗ Failed to load data")
    exit(1)

# 2. Train RF on env_broad_scale (fastest, most balanced)
print("\n2. Training RF model on env_broad_scale...")
X = np.vstack(df["ge_embedding"].values)
y = df["env_broad_scale"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

result = train_rf_model(X_train, y_train, X_test, y_test)
y_pred = result["y_test_pred"]

print(f"   Test set size: {len(X_test)}")
print(f"   Number of classes: {result['n_classes']}")
print(f"   Test accuracy: {result['test_accuracy']:.4f}")
print(f"   Macro avg F1: {result['class_report']['macro avg']['f1-score']:.4f}")
print(f"   Weighted avg F1: {result['class_report']['weighted avg']['f1-score']:.4f}")

# 3. Load ENVO hierarchy
print("\n3. Loading ENVO ontology...")
try:
    ontology = ENVOHierarchy()
    print("   ✓ ENVO loaded via oaklib")
except Exception as e:
    print(f"   ✗ Failed to load ENVO: {e}")
    exit(1)

# 4. Compute hierarchical scores
print("\n4. Computing hierarchical scores on test set...")
hierarchical_scores = [
    hierarchical_score(pred, true, ontology) for pred, true in zip(y_pred, y_test)
]

exact_match_accuracy = np.mean(y_pred == y_test)
hierarchical_acc = np.mean(hierarchical_scores)

print(f"   Exact match accuracy: {exact_match_accuracy:.4f}")
print(f"   Hierarchical accuracy: {hierarchical_acc:.4f}")
print(f"   Improvement: +{(hierarchical_acc - exact_match_accuracy):.4f}")

# 5. Per-class F1 comparison
print("\n5. Per-class performance (top 10 classes)...")
class_report = result["class_report"]

class_metrics = []
for class_label, metrics in class_report.items():
    if class_label not in ["accuracy", "macro avg", "weighted avg"]:
        class_metrics.append(
            {
                "Class": class_label,
                "F1": metrics["f1-score"],
                "Support": int(metrics["support"]),
            }
        )

class_df = pd.DataFrame(class_metrics).sort_values("F1", ascending=False)

print("\n   Class | F1 Score | Support")
print("   ------|----------|--------")
for _, row in class_df.head(10).iterrows():
    print(f"   {row['Class']:30s} | {row['F1']:8.4f} | {row['Support']:7d}")

# 6. Distribution of hierarchical scores
print("\n6. Distribution of hierarchical scores:")
score_bins = {
    "Exact (1.0)": np.sum(np.array(hierarchical_scores) == 1.0),
    "1-hop (0.75)": np.sum(np.array(hierarchical_scores) == 0.75),
    "2-hop (0.50)": np.sum(np.array(hierarchical_scores) == 0.50),
    "3+-hop (0.25)": np.sum(np.array(hierarchical_scores) == 0.25),
    "Unrelated (0.0)": np.sum(np.array(hierarchical_scores) == 0.0),
}

total_predictions = len(hierarchical_scores)
for bin_name, count in score_bins.items():
    pct = (count / total_predictions) * 100
    print(f"   {bin_name:20s}: {count:5d} ({pct:6.2f}%)")

# 7. Show a few examples
print("\n7. Example predictions (5 random samples):")
print("\n   Predicted | True | Exact? | Hier. Score | Distance")
print("   ----------|------|--------|-------------|---------")

sample_indices = np.random.choice(len(y_test), 5, replace=False)
for idx in sample_indices:
    pred = y_pred[idx]
    true = y_test[idx]
    exact = "✓" if pred == true else "✗"
    h_score = hierarchical_scores[idx]
    dist = ontology.distance(pred, true)
    print(f"   {pred:10s} | {true:4s} | {exact:6s} | {h_score:11.2f} | {dist:8.0f}")

print("\n" + "=" * 80)
print("✓ Test complete!")
print("=" * 80 + "\n")
