"""Random Forest analysis for ENVO prediction from Google Earth embeddings.

This module provides clean, modular functions for training and evaluating
Random Forest classifiers that predict ENVO environmental triad terms
from satellite imagery embeddings.
"""

import ast
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

# ENVO scales to analyze
ENVO_SCALES = ["env_broad_scale", "env_local_scale", "env_medium"]

# Random seed for reproducibility
RANDOM_STATE = 42


def parse_embedding(embedding_str) -> Optional[np.ndarray]:
    """Parse embedding string to numpy array.

    Args:
        embedding_str: String representation of embedding list or actual list

    Returns:
        numpy array of embedding values, or None if parsing fails

    Examples:
        >>> emb = parse_embedding("[1.0, 2.0, 3.0]")
        >>> emb.shape
        (3,)
        >>> parse_embedding("invalid") is None
        True
    """
    try:
        if isinstance(embedding_str, str):
            embedding_list = ast.literal_eval(embedding_str)
        else:
            embedding_list = embedding_str
        return np.array(embedding_list, dtype=np.float32)
    except Exception:
        return None


def load_source_data(
    file_path: Path, source_name: str, deduplicate: bool = True
) -> Optional[pd.DataFrame]:
    """Load and prepare data for a single source.

    Args:
        file_path: Path to the CSV file
        source_name: Name of the data source (GOLD/NCBI/NMDC)
        deduplicate: Whether to remove exact duplicates

    Returns:
        Prepared DataFrame with parsed embeddings, or None if loading fails

    Examples:
        >>> # df = load_source_data(Path("data.csv"), "NMDC")  # doctest: +SKIP
        >>> pass
    """
    print(f"\n{'=' * 60}")
    print(f"Loading {source_name}")
    print(f"{'=' * 60}")

    try:
        # Load data
        df = pd.read_csv(file_path)
        print(f"  Loaded: {len(df):,} rows")

        # Deduplicate if requested
        if deduplicate:
            initial_count = len(df)
            df = df.drop_duplicates(
                subset=[
                    "latitude",
                    "longitude",
                    "collection_date",
                    "env_broad_scale",
                    "env_local_scale",
                    "env_medium",
                ],
                keep="first",
            ).copy()
            removed = initial_count - len(df)
            pct_removed = removed / initial_count * 100
            print(
                f"  Deduped: {len(df):,} rows (removed {removed:,} = {pct_removed:.1f}%)"
            )

            # Report if significant deduplication occurred
            if pct_removed > 20:
                print(
                    f"  ⚠️  NOTE: Removed {pct_removed:.1f}% exact duplicates - "
                    "this indicates significant pseudo-replication"
                )

        # Filter to complete data
        df_clean = df[
            df["google_earth_embeddings"].notna()
            & df["env_broad_scale"].notna()
            & df["env_local_scale"].notna()
            & df["env_medium"].notna()
        ].copy()
        print(f"  Complete data: {len(df_clean):,} rows")

        # Parse embeddings
        df_clean["ge_embedding"] = df_clean["google_earth_embeddings"].apply(
            parse_embedding  # type: ignore[arg-type]
        )
        df_clean = df_clean[df_clean["ge_embedding"].notna()].copy()
        print(f"  Valid embeddings: {len(df_clean):,} rows")

        if len(df_clean) == 0:
            print(f"  ✗ No valid data for {source_name}")
            return None

        # Print class counts
        print("\n  ENVO scale class counts:")
        for scale in ENVO_SCALES:
            n_classes = df_clean[scale].nunique()
            print(f"    {scale:20s}: {n_classes:4d} classes")

        return df_clean

    except Exception as e:
        print(f"  ✗ Error loading {source_name}: {e}")
        return None


def train_rf_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 10,
) -> Dict:
    """Train Random Forest classifier and evaluate.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_estimators: Number of trees in forest
        max_depth: Maximum tree depth

    Returns:
        Dictionary with model and performance metrics

    Examples:
        >>> X = np.random.rand(100, 10)  # doctest: +SKIP
        >>> y = np.random.randint(0, 3, 100)  # doctest: +SKIP
        >>> X_train, X_test = X[:80], X[80:]  # doctest: +SKIP
        >>> y_train, y_test = y[:80], y[80:]  # doctest: +SKIP
        >>> result = train_rf_model(X_train, y_train, X_test, y_test)  # doctest: +SKIP
        >>> 'test_accuracy' in result  # doctest: +SKIP
        True
    """
    # Train model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Cross-validation on training set
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy")

    return {
        "model": rf,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "overfitting": train_acc - test_acc,
        "n_classes": len(np.unique(y_train)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def filter_rare_classes(
    df: pd.DataFrame, scale: str, min_samples: int = 5, report_removed: bool = True
) -> pd.DataFrame:
    """Filter out classes with too few samples.

    Args:
        df: DataFrame with ENVO labels
        scale: ENVO scale column name
        min_samples: Minimum samples required per class
        report_removed: Whether to print details of removed classes

    Returns:
        Filtered DataFrame with only classes having >= min_samples

    Examples:
        >>> # df_filtered = filter_rare_classes(df, "env_medium", min_samples=5)  # doctest: +SKIP
        >>> pass
    """
    # Count samples per class
    class_counts = df[scale].value_counts()

    # Find classes with enough samples
    valid_classes = class_counts[class_counts >= min_samples].index
    removed_classes = class_counts[class_counts < min_samples]

    # Filter dataframe
    df_filtered = df[df[scale].isin(valid_classes)].copy()

    n_removed_classes = len(removed_classes)
    n_removed_samples = len(df) - len(df_filtered)

    if n_removed_classes > 0:
        print(
            f"    Filtered: removed {n_removed_classes} rare classes "
            f"({n_removed_samples} samples with <{min_samples} per class)"
        )

        # Report which classes were removed
        if report_removed and n_removed_classes > 0:
            print(f"    Removed classes ({n_removed_classes}):")
            for term, count in removed_classes.head(10).items():
                print(f"      - {term}: {count} samples")
            if n_removed_classes > 10:
                print(f"      ... and {n_removed_classes - 10} more")

    return df_filtered


def analyze_source(
    df: pd.DataFrame,
    source_name: str,
    test_size: float = 0.2,
    min_samples_per_class: int = 5,
) -> Dict[str, Dict]:
    """Train RF models for all ENVO scales for a single data source.

    Args:
        df: DataFrame with embeddings and ENVO labels
        source_name: Name of the data source
        test_size: Fraction of data to use for testing
        min_samples_per_class: Minimum samples required per class (to avoid CV warnings)

    Returns:
        Dictionary mapping scale names to result dictionaries

    Examples:
        >>> # results = analyze_source(df, "NMDC")  # doctest: +SKIP
        >>> pass
    """
    print(f"\n{'-' * 60}")
    print(f"Training models: {source_name}")
    print(f"{'-' * 60}")

    results = {}

    for scale in ENVO_SCALES:
        print(f"\n  {scale}...")

        # Filter rare classes
        df_scale = filter_rare_classes(df, scale, min_samples_per_class)

        if len(df_scale) == 0:
            print("    ✗ No data remaining after filtering")
            continue

        # Prepare feature matrix and target
        X = np.vstack(df_scale["ge_embedding"].values)  # type: ignore[call-overload]
        y = df_scale[scale].values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )

        # Train and evaluate
        result = train_rf_model(X_train, y_train, X_test, y_test)
        results[scale] = result

        # Print compact results
        print(
            f"    Classes: {result['n_classes']:3d} | "
            f"Test acc: {result['test_accuracy']:.3f} | "
            f"CV: {result['cv_mean']:.3f}±{result['cv_std'] * 2:.3f} | "
            f"Overfit: {result['overfitting']:+.3f}"
        )

    return results


def create_comparison_table(all_results: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
    """Create comparison table from all results.

    Args:
        all_results: Nested dict of {source: {scale: result}}

    Returns:
        DataFrame with comparison data

    Examples:
        >>> # results = {"NMDC": {"env_broad_scale": {...}}}  # doctest: +SKIP
        >>> # df = create_comparison_table(results)  # doctest: +SKIP
        >>> pass
    """
    rows = []

    for source in all_results:
        for scale in all_results[source]:
            r = all_results[source][scale]
            rows.append(
                {
                    "Source": source,
                    "Scale": scale.replace("env_", ""),
                    "Classes": r["n_classes"],
                    "Samples": r["n_train"] + r["n_test"],
                    "Test_Acc": r["test_accuracy"],
                    "CV_Mean": r["cv_mean"],
                    "CV_Std": r["cv_std"],
                    "Overfitting": r["overfitting"],
                }
            )

    return pd.DataFrame(rows)


def print_summary(comparison_df: pd.DataFrame):
    """Print actionable summary of results.

    Args:
        comparison_df: Comparison DataFrame from create_comparison_table

    Examples:
        >>> # print_summary(comparison_df)  # doctest: +SKIP
        >>> pass
    """
    print(f"\n\n{'=' * 80}")
    print("SUMMARY: Random Forest ENVO Prediction Performance")
    print(f"{'=' * 80}\n")

    # Overall stats
    mean_acc = comparison_df["Test_Acc"].mean()
    print(f"Overall mean test accuracy: {mean_acc:.3f}\n")

    # Best/worst
    best = comparison_df.loc[comparison_df["Test_Acc"].idxmax()]
    worst = comparison_df.loc[comparison_df["Test_Acc"].idxmin()]
    print(f"Best:  {best['Source']:6s} {best['Scale']:12s} = {best['Test_Acc']:.3f}")
    print(
        f"Worst: {worst['Source']:6s} {worst['Scale']:12s} = {worst['Test_Acc']:.3f}\n"
    )

    # By source
    print("Performance by source:")
    for source in comparison_df["Source"].unique():
        source_data = comparison_df[comparison_df["Source"] == source]
        print(
            f"  {source:6s}: {source_data['Test_Acc'].mean():.3f} avg, "
            f"{source_data['Overfitting'].mean():+.3f} overfit"
        )

    # By scale
    print("\nPerformance by scale:")
    for scale in comparison_df["Scale"].unique():
        scale_data = comparison_df[comparison_df["Scale"] == scale]
        print(
            f"  {scale:12s}: {scale_data['Test_Acc'].mean():.3f} avg, "
            f"{scale_data['Classes'].mean():.0f} avg classes"
        )

    # Actionable insights
    print(f"\n{'=' * 80}")
    print("ACTIONABLE INSIGHTS")
    print(f"{'=' * 80}\n")

    if mean_acc > 0.8:
        print("✓ Satellite embeddings have STRONG predictive power for ENVO terms")
    elif mean_acc > 0.6:
        print("~ Satellite embeddings have MODERATE predictive power")
    else:
        print("✗ Satellite embeddings have LIMITED predictive power")

    # Check overfitting
    max_overfit = comparison_df["Overfitting"].max()
    if max_overfit > 0.15:
        print(
            f"⚠️  High overfitting detected (max={max_overfit:.3f}) - "
            "consider regularization"
        )
    elif max_overfit > 0.1:
        print(f"⚠️  Moderate overfitting (max={max_overfit:.3f})")
    else:
        print(f"✓ Good generalization (max overfit={max_overfit:.3f})")

    # Check deduplication impact if multiple sources
    if len(comparison_df["Source"].unique()) > 1:
        source_var = comparison_df.groupby("Source")["Test_Acc"].mean().std()
        if source_var > 0.1:
            print(
                f"\n⚠️  Large variation across sources (std={source_var:.3f}) - "
                "data quality differs"
            )
