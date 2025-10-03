"""CLI interface for env-embeddings."""

from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from env_embeddings.earth_engine import initialize_ee, get_embedding
from env_embeddings.sample_processor import load_sample_data, process_samples_batch

# from .embeddings import (
#     compute_embedding,
#     compute_embeddings_batch,
#     cosine_similarity,
#     get_similarity_matrix,
#     find_most_similar,
# )

app = typer.Typer(
    help="env-embeddings: Simple experiment to compare ENVO similarity to google embedding cosine similarity"
)


# TODO: Uncomment these commands when embeddings module is implemented
# @app.command()
# def embed(...):
#     """Compute embedding for a single text."""
#     pass

# @app.command()
# def batch_embed(...):
#     """Compute embeddings for multiple texts from a file."""
#     pass

# @app.command()
# def similarity(...):
#     """Compute cosine similarity between two texts."""
#     pass

# @app.command()
# def similarity_matrix(...):
#     """Compute pairwise similarity matrix for texts."""
#     pass

# @app.command()
# def search(...):
#     """Find most similar texts in a corpus."""
#     pass


@app.command()
def init_ee(
    project: Annotated[
        Optional[str], typer.Option(help="Google Cloud project ID (optional)")
    ] = None,
):
    """Initialize Google Earth Engine authentication."""
    try:
        initialize_ee(project=project)
        if project:
            typer.echo(
                f"Google Earth Engine initialized successfully with project: {project}"
            )
        else:
            typer.echo(
                "Google Earth Engine initialized successfully with default credentials"
            )
    except Exception as e:
        typer.echo(f"Error initializing Earth Engine: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def embedding(
    lat: Annotated[float, typer.Option(help="Latitude coordinate")],
    lon: Annotated[float, typer.Option(help="Longitude coordinate")],
    year: Annotated[int, typer.Option(help="Year for the embedding")] = 2024,
    project: Annotated[
        Optional[str], typer.Option(help="Google Cloud project ID (optional)")
    ] = None,
):
    """Get Google Earth Engine embedding for given coordinates and year."""
    try:
        vec = get_embedding(lat, lon, year, project)
        typer.echo(f"Embedding vector for ({lat}, {lon}) in {year}:")
        typer.echo(f"Length: {len(vec)}")
        typer.echo(f"Vector: {vec}")
    except Exception as e:
        typer.echo(f"Error getting embedding: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def process_samples(
    tsv_file: Annotated[Path, typer.Argument(help="Path to TSV file with sample data")],
    output: Annotated[Path, typer.Option(help="Output file for results")] = Path(
        "data/samples_with_embeddings.tsv"
    ),
    max_samples: Annotated[
        Optional[int],
        typer.Option(help="Maximum number of samples to process (for testing)"),
    ] = None,
    project: Annotated[
        str, typer.Option(help="Google Cloud project ID")
    ] = "env-embeddings-2025",
):
    """Process sample data to get Earth Engine embeddings for each sample."""
    # If tsv_file is just a filename, look in data/ directory
    if not tsv_file.exists() and not tsv_file.is_absolute():
        data_path = Path("data") / tsv_file
        if data_path.exists():
            tsv_file = data_path

    if not tsv_file.exists():
        typer.echo(f"Error: {tsv_file} not found", err=True)
        raise typer.Exit(1)

    try:
        # Load and parse sample data
        typer.echo(f"Loading sample data from {tsv_file}...")
        samples = load_sample_data(tsv_file, max_samples)

        if len(samples) == 0:
            typer.echo("No valid samples found with coordinates and dates", err=True)
            raise typer.Exit(1)

        # Process samples to get embeddings
        typer.echo(f"Processing {len(samples)} samples...")
        results = process_samples_batch(samples, project, output)

        typer.echo(f"Successfully processed {len(results)} samples")
        typer.echo(f"Results saved to {output}")

    except Exception as e:
        typer.echo(f"Error processing samples: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def add_embeddings(
    tsv_file: Annotated[
        Path, typer.Argument(help="Path to TSV file to add embeddings to")
    ],
    output: Annotated[
        Optional[Path], typer.Option(help="Output file with embeddings added")
    ] = None,
    max_rows: Annotated[
        Optional[int],
        typer.Option(help="Maximum number of rows to process (for testing)"),
    ] = None,
    project: Annotated[
        str, typer.Option(help="Google Cloud project ID")
    ] = "env-embeddings-2025",
    fallback_year: Annotated[
        int, typer.Option(help="Year to use when original year has no data")
    ] = 2020,
    skip_existing: Annotated[
        bool, typer.Option(help="Skip rows that already have embeddings")
    ] = True,
):
    """Add Google Earth Engine embeddings to existing TSV file."""
    # If tsv_file is just a filename, look in data/ directory
    if not tsv_file.exists() and not tsv_file.is_absolute():
        data_path = Path("data") / tsv_file
        if data_path.exists():
            tsv_file = data_path

    if not tsv_file.exists():
        typer.echo(f"Error: {tsv_file} not found", err=True)
        raise typer.Exit(1)

    if output is None:
        # Default output to data/ directory
        output = Path("data") / f"{tsv_file.stem}_with_embeddings{tsv_file.suffix}"

    try:
        from env_embeddings.sample_processor import add_embeddings_to_tsv

        typer.echo(f"Adding Earth Engine embeddings to {tsv_file}...")
        typer.echo(f"Output will be saved to: {output}")

        success_count = add_embeddings_to_tsv(
            tsv_file=tsv_file,
            output_file=output,
            max_rows=max_rows,
            project=project,
            fallback_year=fallback_year,
            skip_existing=skip_existing,
        )

        typer.echo(f"Successfully added embeddings to {success_count} rows")
        typer.echo(f"Results saved to {output}")

    except Exception as e:
        typer.echo(f"Error adding embeddings: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def add_envo_embeddings(
    tsv_file: Annotated[
        Path, typer.Argument(help="Path to TSV file to add ENVO embeddings to")
    ],
    output: Annotated[
        Optional[Path], typer.Option(help="Output file with ENVO embeddings added")
    ] = None,
    max_rows: Annotated[
        Optional[int],
        typer.Option(help="Maximum number of rows to process (for testing)"),
    ] = None,
    skip_existing: Annotated[
        bool, typer.Option(help="Skip rows that already have ENVO embeddings")
    ] = True,
):
    """Add ENVO embeddings to existing TSV file using OLS."""
    # If tsv_file is just a filename, look in data/ directory
    if not tsv_file.exists() and not tsv_file.is_absolute():
        data_path = Path("data") / tsv_file
        if data_path.exists():
            tsv_file = data_path

    if not tsv_file.exists():
        typer.echo(f"Error: {tsv_file} not found", err=True)
        raise typer.Exit(1)

    if output is None:
        # Default output to data/ directory
        output = Path("data") / f"{tsv_file.stem}_with_envo_embeddings{tsv_file.suffix}"

    try:
        from env_embeddings.sample_processor import add_envo_embeddings_to_tsv

        typer.echo(f"Adding ENVO embeddings to {tsv_file}...")
        typer.echo(f"Output will be saved to: {output}")

        success_count = add_envo_embeddings_to_tsv(
            tsv_file=tsv_file,
            output_file=output,
            max_rows=max_rows,
            skip_existing=skip_existing,
        )

        typer.echo(f"Successfully added ENVO embeddings to {success_count} rows")
        typer.echo(f"Results saved to {output}")

    except Exception as e:
        typer.echo(f"Error adding ENVO embeddings: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def add_google_embeddings_csv(
    csv_file: Annotated[Path, typer.Argument(help="Path to normalized CSV file")],
    output: Annotated[
        Optional[Path], typer.Option(help="Output file with Google Earth embeddings")
    ] = None,
    max_rows: Annotated[
        Optional[int],
        typer.Option(help="Maximum number of rows to process (all if None)"),
    ] = None,
    project: Annotated[
        str, typer.Option(help="Google Cloud project ID")
    ] = "env-embeddings-2025",
    fallback_year: Annotated[
        int, typer.Option(help="Year to use when original year has no data")
    ] = 2020,
    skip_existing: Annotated[
        bool, typer.Option(help="Skip rows that already have embeddings")
    ] = True,
    random_sample: Annotated[
        bool,
        typer.Option(
            "--random/--sequential",
            help="Random sample (default, avoids bias) vs sequential (first N rows)",
        ),
    ] = True,
):
    """Add Google Earth Engine embeddings to Mark's normalized CSV file.

    Expects CSV with columns: collection_date, latitude, longitude

    SAMPLING MODES:
      --random (default):  Randomly sample max_rows for representative coverage
                          ✅ Recommended for exploration and statistical validity
                          ✅ Avoids geographic/temporal/bioproject bias
                          Uses seed=42 for reproducibility

      --sequential:        Take first max_rows in order
                          ✅ Useful for resuming failed jobs
                          ⚠️  May introduce bias if data is ordered
    """
    # If csv_file is just a filename, look in data/ directory
    if not csv_file.exists() and not csv_file.is_absolute():
        data_path = Path("data") / csv_file
        if data_path.exists():
            csv_file = data_path

    if not csv_file.exists():
        typer.echo(f"Error: {csv_file} not found", err=True)
        raise typer.Exit(1)

    if output is None:
        # Default output to data/ directory
        output = Path("data") / f"{csv_file.stem}_with_google_embeddings.csv"

    try:
        from env_embeddings.sample_processor import add_google_earth_embeddings_to_csv

        typer.echo(f"Adding Google Earth Engine embeddings to {csv_file}...")
        typer.echo(
            f"Sampling mode: {'RANDOM (seed=42)' if random_sample else 'SEQUENTIAL (first N)'}"
        )
        typer.echo(f"Output will be saved to: {output}")

        success_count = add_google_earth_embeddings_to_csv(
            csv_file=csv_file,
            output_file=output,
            max_rows=max_rows,
            project=project,
            fallback_year=fallback_year,
            skip_existing=skip_existing,
            random_sample=random_sample,
        )

        typer.echo(
            f"Successfully added Google Earth embeddings to {success_count} rows"
        )
        typer.echo(f"Results saved to {output}")

    except Exception as e:
        typer.echo(f"Error adding Google Earth embeddings: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def add_envo_embeddings_csv(
    csv_file: Annotated[Path, typer.Argument(help="Path to normalized CSV file")],
    output: Annotated[
        Optional[Path], typer.Option(help="Output file with ENVO embeddings")
    ] = None,
    max_rows: Annotated[
        Optional[int],
        typer.Option(help="Maximum number of rows to process (all if None)"),
    ] = None,
    skip_existing: Annotated[
        bool, typer.Option(help="Skip rows that already have embeddings")
    ] = True,
    random_sample: Annotated[
        bool,
        typer.Option(
            "--random/--sequential",
            help="Random sample (default, avoids bias) vs sequential (first N rows)",
        ),
    ] = True,
):
    """Add ENVO embeddings for all three environmental columns to CSV file.

    Processes env_broad_scale, env_medium, env_local_scale columns.
    Creates three embedding columns with caching to avoid redundant API calls.

    SAMPLING MODES:
      --random (default):  Randomly sample max_rows for representative coverage
                          ✅ Recommended for exploration and statistical validity
                          ✅ Avoids geographic/temporal/bioproject bias
                          Uses seed=42 for reproducibility

      --sequential:        Take first max_rows in order
                          ✅ Useful for resuming failed jobs
                          ⚠️  May introduce bias if data is ordered
    """
    # If csv_file is just a filename, look in data/ directory
    if not csv_file.exists() and not csv_file.is_absolute():
        data_path = Path("data") / csv_file
        if data_path.exists():
            csv_file = data_path

    if not csv_file.exists():
        typer.echo(f"Error: {csv_file} not found", err=True)
        raise typer.Exit(1)

    if output is None:
        # Default output to data/ directory
        output = Path("data") / f"{csv_file.stem}_with_envo_embeddings.csv"

    try:
        from env_embeddings.sample_processor import add_envo_embeddings_to_csv

        typer.echo(f"Adding ENVO embeddings to {csv_file}...")
        typer.echo(
            f"Sampling mode: {'RANDOM (seed=42)' if random_sample else 'SEQUENTIAL (first N)'}"
        )
        typer.echo(f"Output will be saved to: {output}")

        success_count = add_envo_embeddings_to_csv(
            csv_file=csv_file,
            output_file=output,
            max_rows=max_rows,
            skip_existing=skip_existing,
            random_sample=random_sample,
        )

        typer.echo(f"Successfully added ENVO embeddings to {success_count} rows")
        typer.echo(f"Results saved to {output}")

    except Exception as e:
        typer.echo(f"Error adding ENVO embeddings: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def rank_bioprojects(
    csv_file: Annotated[Path, typer.Argument(help="Path to CSV file with embeddings")],
    output: Annotated[
        Optional[Path], typer.Option(help="Output file for rankings")
    ] = None,
    envo_scale: Annotated[
        str, typer.Option(help="ENVO scale to use for analysis")
    ] = "envo_broad_scale",
    mongo_host: Annotated[str, typer.Option(help="MongoDB host")] = "localhost",
    mongo_port: Annotated[int, typer.Option(help="MongoDB port")] = 27017,
):
    """Rank bioprojects by annotation quality based on GE vs ENVO agreement.

    Analyzes how well each bioproject's MIxS metadata annotations match satellite
    imagery by computing pairwise similarities within each bioproject.

    REQUIREMENTS:
      - CSV file with both google_earth_embeddings and envo_*_embedding columns
      - MongoDB running at mongo_host:mongo_port with ncbi_metadata.sra_biosamples_bioprojects

    ENVO SCALE OPTIONS:
      - envo_broad_scale (default): Broad environmental context
      - envo_local_scale: Fine-grained local features
      - envo_medium: Intermediate environmental medium

    OUTPUT METRICS:
      - quality_score: 0-1, higher = better annotation quality
      - mean_disagreement: Average |GE_sim - ENVO_sim|
      - pearson_r, spearman_r: Correlation within bioproject
      - n_samples, n_pairs: Sample and pair counts
    """
    # If csv_file is just a filename, look in data/ directory
    if not csv_file.exists() and not csv_file.is_absolute():
        data_path = Path("data") / csv_file
        if data_path.exists():
            csv_file = data_path

    if not csv_file.exists():
        typer.echo(f"Error: {csv_file} not found", err=True)
        raise typer.Exit(1)

    if output is None:
        # Default output to data/ directory
        output = Path("data") / f"{csv_file.stem}_bioproject_rankings.csv"

    # Validate envo_scale
    valid_scales = ["envo_broad_scale", "envo_local_scale", "envo_medium"]
    if envo_scale not in valid_scales:
        typer.echo(f"Error: envo_scale must be one of {valid_scales}", err=True)
        raise typer.Exit(1)

    try:
        from env_embeddings.bioproject_quality import rank_bioprojects as rank_fn

        typer.echo(f"Ranking bioprojects from {csv_file}...")
        typer.echo(f"Using ENVO scale: {envo_scale}")
        typer.echo(f"MongoDB: {mongo_host}:{mongo_port}")
        typer.echo(f"Output will be saved to: {output}")
        typer.echo()

        results = rank_fn(
            csv_file=csv_file,
            mongo_host=mongo_host,
            mongo_port=mongo_port,
            output_file=output,
            envo_scale=envo_scale,
        )

        typer.echo(f"\n✅ Successfully ranked {len(results)} bioprojects")
        typer.echo(f"Results saved to {output}")

    except Exception as e:
        typer.echo(f"Error ranking bioprojects: {e}", err=True)
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
