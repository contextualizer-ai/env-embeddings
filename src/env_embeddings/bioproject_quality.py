"""Analyze bioproject quality based on GE vs ENVO agreement.

This module connects to MongoDB to get bioproject associations,
then calculates quality metrics for each bioproject based on how well
their MIxS annotations match satellite embeddings.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
import ast


def parse_embedding(embedding_str) -> Optional[np.ndarray]:  # type: ignore[return]
    """Parse embedding string to numpy array."""
    try:
        if isinstance(embedding_str, str):
            embedding_list = ast.literal_eval(embedding_str)
        else:
            embedding_list = embedding_str
        return np.array(embedding_list, dtype=np.float32)
    except Exception:
        return None


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    if emb1 is None or emb2 is None:
        return np.nan

    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)

    return cosine_similarity(emb1, emb2)[0, 0]


def get_bioproject_mapping(
    mongo_host: str = "localhost",
    mongo_port: int = 27017,
    database: str = "ncbi_metadata",
    collection: str = "sra_biosamples_bioprojects",
) -> Dict[str, str]:
    """Get biosample -> bioproject mapping from MongoDB.

    Args:
        mongo_host: MongoDB host
        mongo_port: MongoDB port
        database: Database name
        collection: Collection name

    Returns:
        Dictionary mapping biosample accession to bioproject accession
    """
    client: MongoClient = MongoClient(mongo_host, mongo_port)  # type: ignore[type-arg]
    db = client[database]
    coll = db[collection]

    mapping = {}
    for doc in coll.find():
        biosample_acc = doc.get("biosample_accession")
        bioproject_acc = doc.get("bioproject_accession")

        if biosample_acc and bioproject_acc:
            mapping[biosample_acc] = bioproject_acc

    print(f"Loaded {len(mapping)} biosample -> bioproject mappings from MongoDB")
    return mapping


def calculate_intra_bioproject_agreement(
    df: pd.DataFrame,
    bioproject_mapping: Dict[str, str],
    envo_column: str = "envo_broad_scale",
) -> pd.DataFrame:
    """Calculate agreement metrics within each bioproject.

    For each bioproject, calculate pairwise similarities between all samples
    and compute mean agreement between GE and ENVO.

    Args:
        df: DataFrame with embeddings (must have ge_embedding and envo_*_embedding columns)
        bioproject_mapping: Dict mapping biosample accession to bioproject
        envo_column: Which ENVO scale to use ('envo_broad_scale', 'envo_local_scale', 'envo_medium')

    Returns:
        DataFrame with bioproject quality metrics
    """
    # Add bioproject column to dataframe
    df["bioproject"] = df["accession"].map(bioproject_mapping)

    # Filter to only samples with bioproject info
    df_with_project = df[df["bioproject"].notna()].copy()

    print(f"Samples with bioproject info: {len(df_with_project)} / {len(df)}")

    # Parse embeddings
    df_with_project["ge_emb"] = df_with_project["google_earth_embeddings"].apply(
        parse_embedding  # type: ignore[arg-type,assignment]
    )

    envo_col_map = {
        "envo_broad_scale": "envo_broad_scale_embedding",
        "envo_local_scale": "envo_local_scale_embedding",
        "envo_medium": "envo_medium_embedding",
    }
    envo_embedding_col = envo_col_map.get(envo_column, "envo_broad_scale_embedding")

    df_with_project["envo_emb"] = df_with_project[envo_embedding_col].apply(
        parse_embedding  # type: ignore[arg-type,assignment]
    )

    # Filter out parsing failures
    df_clean = df_with_project[
        df_with_project["ge_emb"].notna() & df_with_project["envo_emb"].notna()
    ].copy()

    print(f"Samples with valid embeddings: {len(df_clean)}")

    # Group by bioproject
    bioproject_metrics = []

    for bioproject, group in df_clean.groupby("bioproject"):
        n_samples = len(group)

        if n_samples < 2:
            # Need at least 2 samples for pairwise comparison
            continue

        # Calculate all pairwise similarities
        ge_similarities = []
        envo_similarities = []

        samples = list(group.iterrows())
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                _, sample1 = samples[i]
                _, sample2 = samples[j]

                ge_sim = compute_cosine_similarity(sample1["ge_emb"], sample2["ge_emb"])
                envo_sim = compute_cosine_similarity(
                    sample1["envo_emb"], sample2["envo_emb"]
                )

                if not np.isnan(ge_sim) and not np.isnan(envo_sim):
                    ge_similarities.append(ge_sim)
                    envo_similarities.append(envo_sim)

        if not ge_similarities:
            continue

        # Calculate metrics
        n_pairs = len(ge_similarities)

        # Mean similarities
        mean_ge_sim = np.mean(ge_similarities)
        mean_envo_sim = np.mean(envo_similarities)

        # Correlation between GE and ENVO (within this bioproject)
        from scipy.stats import pearsonr, spearmanr

        if len(ge_similarities) > 2:  # Need at least 3 pairs for correlation
            pearson_r, pearson_p = pearsonr(ge_similarities, envo_similarities)
            spearman_r, spearman_p = spearmanr(ge_similarities, envo_similarities)
        else:
            pearson_r = pearson_p = spearman_r = spearman_p = float(np.nan)  # type: ignore[assignment]

        # Agreement score: absolute difference between similarities
        # Low difference = good agreement (both high or both low)
        disagreements = np.abs(np.array(ge_similarities) - np.array(envo_similarities))
        mean_disagreement = np.mean(disagreements)

        # Quality score: inverse of disagreement (higher is better)
        quality_score = 1.0 - mean_disagreement

        bioproject_metrics.append(
            {
                "bioproject": bioproject,
                "n_samples": n_samples,
                "n_pairs": n_pairs,
                "mean_ge_similarity": mean_ge_sim,
                "mean_envo_similarity": mean_envo_sim,
                "mean_disagreement": mean_disagreement,
                "quality_score": quality_score,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
        )

    results_df = pd.DataFrame(bioproject_metrics)
    results_df = results_df.sort_values("quality_score", ascending=False)

    return results_df


def rank_bioprojects(
    csv_file: Path,
    mongo_host: str = "localhost",
    mongo_port: int = 27017,
    output_file: Optional[Path] = None,
    envo_scale: str = "envo_broad_scale",
) -> pd.DataFrame:
    """Rank bioprojects by annotation quality.

    Args:
        csv_file: CSV file with embeddings
        mongo_host: MongoDB host
        mongo_port: MongoDB port
        output_file: Optional path to save ranked results
        envo_scale: Which ENVO scale to use

    Returns:
        DataFrame with bioproject rankings
    """
    # Load embeddings
    print(f"Loading embeddings from {csv_file}")
    df = pd.read_csv(csv_file, low_memory=False)

    # Get bioproject mapping
    print("Connecting to MongoDB...")
    bioproject_mapping = get_bioproject_mapping(mongo_host, mongo_port)

    # Calculate quality metrics
    print(f"Calculating quality metrics using {envo_scale}...")
    results = calculate_intra_bioproject_agreement(df, bioproject_mapping, envo_scale)

    # Print summary
    print(f"\n{'=' * 80}")
    print("BIOPROJECT QUALITY RANKING")
    print(f"{'=' * 80}")
    print(f"Total bioprojects analyzed: {len(results)}")
    print(f"Using ENVO scale: {envo_scale}")
    print()

    print("Top 10 highest quality bioprojects:")
    print(
        results[
            [
                "bioproject",
                "n_samples",
                "quality_score",
                "mean_disagreement",
                "spearman_r",
            ]
        ]
        .head(10)
        .to_string()
    )

    print("\n\nTop 10 lowest quality bioprojects:")
    print(
        results[
            [
                "bioproject",
                "n_samples",
                "quality_score",
                "mean_disagreement",
                "spearman_r",
            ]
        ]
        .tail(10)
        .to_string()
    )

    # Save if requested
    if output_file:
        results.to_csv(output_file, index=False)
        print(f"\nSaved results to {output_file}")

    return results


def main():
    """Example usage."""
    from pathlib import Path

    csv_file = Path("data/biosamples_with_embeddings.csv")
    output_file = Path("data/bioproject_quality_rankings.csv")

    results = rank_bioprojects(
        csv_file=csv_file, output_file=output_file, envo_scale="envo_broad_scale"
    )

    return results


if __name__ == "__main__":
    main()
