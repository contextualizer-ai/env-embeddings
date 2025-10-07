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
def normalize_biosamples(
    input_file: Annotated[
        Path, typer.Argument(help="Path to CSV or TSV file with raw format")
    ],
    output: Annotated[
        Optional[Path], typer.Option(help="Output file with normalized data")
    ] = None,
):
    """Normalize biosample data - parse coordinates and dates without fetching embeddings.

    Converts raw format to normalized format:
        Input:  collection_date_raw, lat_lon_raw
        Output: collection_date (year), latitude, longitude (numeric)

    Useful for:
        - Pre-processing NCBI/GOLD/NMDC extracts before batch embedding retrieval
        - Inspecting parsed data before API calls
        - Debugging coordinate/date parsing issues

    EXAMPLE:
        uv run env-embeddings normalize-biosamples data/ncbi_raw.tsv
    """
    # If input_file is just a filename, look in data/ directory
    if not input_file.exists() and not input_file.is_absolute():
        data_path = Path("data") / input_file
        if data_path.exists():
            input_file = data_path

    if not input_file.exists():
        typer.echo(f"Error: {input_file} not found", err=True)
        raise typer.Exit(1)

    if output is None:
        # Default output to data/ directory, preserve extension
        output = Path("data") / f"{input_file.stem}_normalized{input_file.suffix}"

    try:
        from env_embeddings.sample_processor import normalize_biosamples as normalize_fn

        typer.echo(f"Normalizing biosample data from {input_file}...")
        typer.echo(f"Output will be saved to: {output}")

        success_count = normalize_fn(
            input_file=input_file,
            output_file=output,
        )

        typer.echo(
            f"\n✓ Successfully normalized {success_count} rows with complete data"
        )
        typer.echo(f"Results saved to {output}")

    except Exception as e:
        typer.echo(f"Error normalizing biosamples: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def add_google_embeddings(
    input_file: Annotated[Path, typer.Argument(help="Path to CSV or TSV file")],
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
    batch_size: Annotated[
        Optional[int],
        typer.Option(
            help="Batch size for Earth Engine API calls (500 recommended, None for one-by-one)"
        ),
    ] = None,
):
    """Add Google Earth Engine embeddings - auto-detects format (CSV/TSV, raw/normalized).

    SUPPORTED FORMATS:

    1. Raw format (from SQL extract):
       - CSV or TSV with: accession, collection_date_raw, lat_lon_raw,
         env_broad_scale, env_local_scale, env_medium
       - Automatically parses coordinates and dates

    2. Normalized format:
       - CSV or TSV with: accession, collection_date, latitude, longitude,
         env_broad_scale, env_local_scale, env_medium
       - Uses values directly

    File type (.csv or .tsv) is auto-detected from extension.

    SAMPLING MODES:
      --random (default):  Randomly sample max_rows for representative coverage
                          ✅ Recommended for exploration and statistical validity
                          ✅ Avoids geographic/temporal/bioproject bias
                          Uses seed=42 for reproducibility

      --sequential:        Take first max_rows in order
                          ✅ Useful for resuming failed jobs
                          ⚠️  May introduce bias if data is ordered
    """
    # If input_file is just a filename, look in data/ directory
    if not input_file.exists() and not input_file.is_absolute():
        data_path = Path("data") / input_file
        if data_path.exists():
            input_file = data_path

    if not input_file.exists():
        typer.echo(f"Error: {input_file} not found", err=True)
        raise typer.Exit(1)

    if output is None:
        # Default output to data/ directory, preserve extension
        output = (
            Path("data")
            / f"{input_file.stem}_with_google_embeddings{input_file.suffix}"
        )

    try:
        from env_embeddings.sample_processor import add_google_earth_embeddings_flexible

        typer.echo(f"Adding Google Earth Engine embeddings to {input_file}...")
        typer.echo(
            f"Sampling mode: {'RANDOM (seed=42)' if random_sample else 'SEQUENTIAL (first N)'}"
        )
        typer.echo(f"Output will be saved to: {output}")

        success_count = add_google_earth_embeddings_flexible(
            input_file=input_file,
            output_file=output,
            max_rows=max_rows,
            project=project,
            fallback_year=fallback_year,
            skip_existing=skip_existing,
            random_sample=random_sample,
            batch_size=batch_size,
        )

        typer.echo(
            f"Successfully added Google Earth embeddings to {success_count} rows"
        )
        typer.echo(f"Results saved to {output}")

    except Exception as e:
        typer.echo(f"Error adding Google Earth embeddings: {e}", err=True)
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
    batch_size: Annotated[
        Optional[int],
        typer.Option(
            help="Batch size for Earth Engine API calls (500 recommended, None for one-by-one)"
        ),
    ] = None,
):
    """DEPRECATED: Use add-google-embeddings instead.

    This command is maintained for backwards compatibility.
    """
    return add_google_embeddings(
        input_file=csv_file,
        output=output,
        max_rows=max_rows,
        project=project,
        fallback_year=fallback_year,
        skip_existing=skip_existing,
        random_sample=random_sample,
        batch_size=batch_size,
    )


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
        # Default output with clear naming
        # If input already has "google_earth" or "with_google", append envo
        # Otherwise create complete name
        stem = csv_file.stem
        if "google_earth" in stem or "with_google" in stem:
            output = Path("data") / f"{stem}_with_envo_embeddings.csv"
        else:
            output = Path("data") / f"{stem}_with_google_earth_and_envo_embeddings.csv"

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


@app.command()
def prepare_ee_coords(
    input_file: Annotated[Path, typer.Argument(help="Path to CSV or TSV file")],
    output: Annotated[
        Optional[Path],
        typer.Option(help="Output coordinates CSV (mapping will be *_mapping.csv)"),
    ] = None,
    max_rows: Annotated[
        Optional[int],
        typer.Option(help="Maximum number of rows to process (for testing)"),
    ] = None,
):
    """Prepare optimized coordinates CSV for Earth Engine bulk upload.

    Creates two files:
    1. Coordinates CSV: Unique (lat, lon, year) tuples for EE processing
    2. Mapping CSV: Links sample IDs to coordinates for result expansion

    This reduces duplicate processing - each unique location is processed once,
    then results are expanded back to all samples.

    EXAMPLE:
        uv run env-embeddings prepare-ee-coords data/nmdc.tsv
    """
    # If input_file is just a filename, look in data/ directory
    if not input_file.exists() and not input_file.is_absolute():
        data_path = Path("data") / input_file
        if data_path.exists():
            input_file = data_path

    if not input_file.exists():
        typer.echo(f"Error: {input_file} not found", err=True)
        raise typer.Exit(1)

    if output is None:
        # Default output to data/ directory
        output = Path("data") / f"{input_file.stem}_coords_for_ee.csv"

    try:
        from env_embeddings.sample_processor import prepare_coordinates_for_ee_upload

        typer.echo(f"Preparing coordinates from {input_file}...")

        coords_file, mapping_file = prepare_coordinates_for_ee_upload(
            input_file=input_file,
            output_file=output,
            max_rows=max_rows,
        )

        typer.echo("\n✅ Ready for Earth Engine bulk upload!")
        typer.echo("\nNext steps:")
        typer.echo(f"  1. Upload {coords_file} to Earth Engine")
        typer.echo("  2. Process using Earth Engine script")
        typer.echo(
            f"  3. Download results and merge using mapping file: {mapping_file}"
        )

    except Exception as e:
        typer.echo(f"Error preparing coordinates: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def convert_ee_results(
    input_file: Annotated[
        Path, typer.Argument(help="Path to Earth Engine results CSV")
    ],
    output: Annotated[
        Optional[Path], typer.Option(help="Output file with converted format")
    ] = None,
):
    """Convert Earth Engine results to our embedding format.

    Earth Engine exports embeddings as separate columns (A00, A01, ..., A63).
    This converts them to a single list column for merging.

    EXAMPLE:
        uv run env-embeddings convert-ee-results nmdc_embeddings.csv
    """
    # If input_file is just a filename, look in current directory
    if not input_file.exists() and not input_file.is_absolute():
        data_path = Path("data") / input_file
        if data_path.exists():
            input_file = data_path

    if not input_file.exists():
        typer.echo(f"Error: {input_file} not found", err=True)
        raise typer.Exit(1)

    if output is None:
        # Default output
        output = Path("data") / f"{input_file.stem}_converted.csv"

    try:
        import csv

        typer.echo(f"Converting Earth Engine results from: {input_file}")

        with open(input_file, "r") as f_in:
            reader = csv.DictReader(f_in)

            rows_converted = 0
            rows_failed = 0

            with open(output, "w", newline="") as f_out:
                writer = csv.DictWriter(
                    f_out, fieldnames=["coord_id", "google_earth_embeddings"]
                )
                writer.writeheader()

                for row in reader:
                    coord_id = row["coord_id"]

                    # Extract A00 through A63
                    try:
                        embedding = []
                        for i in range(64):
                            band_name = f"A{str(i).zfill(2)}"
                            value = row.get(band_name)

                            if value is None or value == "":
                                # No coverage for this coordinate
                                raise ValueError(f"Missing band {band_name}")

                            embedding.append(float(value))

                        # Convert to string (same format as our existing embeddings)
                        embedding_str = str(embedding)

                        writer.writerow(
                            {
                                "coord_id": coord_id,
                                "google_earth_embeddings": embedding_str,
                            }
                        )

                        rows_converted += 1

                    except (ValueError, KeyError):
                        # Skip coordinates with no coverage
                        rows_failed += 1
                        continue

        typer.echo("\n✅ Conversion complete:")
        typer.echo(f"  - Converted: {rows_converted} coordinates with embeddings")
        typer.echo(f"  - Skipped: {rows_failed} coordinates (no coverage)")
        typer.echo(f"  - Output: {output}")

    except Exception as e:
        typer.echo(f"Error converting results: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def merge_ee_results_cmd(
    mapping_file: Annotated[Path, typer.Argument(help="Path to mapping CSV")],
    ee_results_file: Annotated[
        Path, typer.Argument(help="Path to Earth Engine results CSV")
    ],
    original_file: Annotated[Path, typer.Argument(help="Path to original input file")],
    output: Annotated[
        Optional[Path], typer.Option(help="Output file with embeddings merged")
    ] = None,
):
    """Merge Earth Engine results back to original samples.

    Uses the mapping file to expand Earth Engine results (unique coordinates)
    back to all original samples.

    EXAMPLE:
        uv run env-embeddings merge-ee-results \\
            data/nmdc_coords_mapping.csv \\
            data/ee_results.csv \\
            data/nmdc.tsv
    """
    if not mapping_file.exists():
        typer.echo(f"Error: {mapping_file} not found", err=True)
        raise typer.Exit(1)

    if not ee_results_file.exists():
        typer.echo(f"Error: {ee_results_file} not found", err=True)
        raise typer.Exit(1)

    if not original_file.exists():
        typer.echo(f"Error: {original_file} not found", err=True)
        raise typer.Exit(1)

    if output is None:
        # Default output with clear naming (google_earth not just "embeddings")
        output = (
            Path("data")
            / f"{original_file.stem}_with_google_earth_embeddings{original_file.suffix}"
        )

    try:
        from env_embeddings.sample_processor import merge_ee_results

        typer.echo("Merging Earth Engine results...")

        success_count = merge_ee_results(
            mapping_file=mapping_file,
            ee_results_file=ee_results_file,
            original_file=original_file,
            output_file=output,
        )

        typer.echo(f"\n✅ Successfully merged {success_count} samples with embeddings")

    except Exception as e:
        typer.echo(f"Error merging results: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def bulk_process_ee(
    coords: Annotated[Path, typer.Argument(help="Path to coordinates CSV")],
    project: Annotated[
        str, typer.Option(help="Google Cloud project ID")
    ] = "env-embeddings-2025",
    export_to: Annotated[str, typer.Option(help="Export destination")] = "drive",
    description: Annotated[
        str, typer.Option(help="Export task description")
    ] = "satellite_embeddings",
    folder: Annotated[
        str, typer.Option(help="Drive folder name (for --export-to drive)")
    ] = "earth_engine_exports",
    bucket: Annotated[
        Optional[str], typer.Option(help="GCS bucket name (for --export-to gcs)")
    ] = None,
    asset: Annotated[
        Optional[str],
        typer.Option(
            help="Use existing Earth Engine asset (e.g., projects/PROJECT/assets/TABLE)"
        ),
    ] = None,
    no_monitor: Annotated[
        bool, typer.Option(help="Start task but don't monitor")
    ] = False,
):
    """Bulk process satellite embeddings using Earth Engine.

    For small datasets (< 2K coords): Uploads coordinates and processes server-side.
    For large datasets (> 10K coords): Use --asset flag to reference pre-uploaded Earth Engine Table asset.

    SMALL DATASET EXAMPLE (NMDC, GOLD):
        uv run env-embeddings bulk-process-ee data/nmdc_coords_for_ee.csv \\
            --description nmdc_embeddings

    LARGE DATASET EXAMPLE (NCBI):
        # First upload CSV to Earth Engine as Table asset via Code Editor
        # Then run:
        uv run env-embeddings bulk-process-ee data/ncbi_coords_for_ee.csv \\
            --asset projects/env-embeddings-2025/assets/ncbi_coords \\
            --description ncbi_embeddings
    """
    if export_to not in ["drive", "gcs"]:
        typer.echo("Error: export-to must be 'drive' or 'gcs'", err=True)
        raise typer.Exit(1)

    if export_to == "gcs" and not bucket:
        typer.echo("Error: --bucket required when using --export-to gcs", err=True)
        raise typer.Exit(1)

    try:
        from env_embeddings.ee_bulk_process import (
            initialize_earth_engine,
            upload_coordinates_to_ee,
            sample_embeddings,
            export_results,
            monitor_task,
        )
        import ee

        typer.echo(f"Initializing Earth Engine with project: {project}")
        initialize_earth_engine(project)

        # Load coordinates - from asset if provided, otherwise upload CSV
        if asset:
            typer.echo(f"\nLoading coordinates from Earth Engine asset: {asset}")
            coords_fc = ee.FeatureCollection(asset)
            count = coords_fc.size().getInfo()
            typer.echo(f"  - Loaded {count} coordinates from asset")
        else:
            # Upload coordinates (original behavior)
            coords_fc = upload_coordinates_to_ee(coords, f"users/tmp/{description}")

        # Process embeddings (server-side)
        typer.echo("\nSetting up server-side sampling...")
        results_fc = sample_embeddings(coords_fc)
        typer.echo("  - Server-side sampling function configured")

        # Export results
        task = export_results(
            results_fc,
            description=description,
            export_to=export_to,
            folder=folder,
            bucket=bucket,
        )

        if not no_monitor:
            typer.echo("\nMonitoring task (checking every 30s)...")
            typer.echo("  Press Ctrl+C to stop monitoring (task will continue running)")
            result = monitor_task(task)

            if result == "COMPLETED":
                typer.echo("✅ Task completed successfully!")
                if export_to == "drive":
                    typer.echo(
                        f"  Download from: https://drive.google.com/#folders/{folder}"
                    )
            elif result == "FAILED":
                typer.echo("❌ Task failed!")
                raise typer.Exit(1)
            elif result is None:
                typer.echo("\n\nMonitoring stopped. Task is still running.")
                typer.echo("Check status at: https://code.earthengine.google.com/tasks")
        else:
            typer.echo("\n✅ Task started (not monitoring)")
            typer.echo("Check status at: https://code.earthengine.google.com/tasks")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
