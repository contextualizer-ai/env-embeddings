"""CLI interface for env-embeddings."""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
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

app = typer.Typer(help="env-embeddings: Simple experiment to compare ENVO similarity to google embedding cosine similarity")


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
    project: Annotated[Optional[str], typer.Option(help="Google Cloud project ID (optional)")] = None
):
    """Initialize Google Earth Engine authentication."""
    try:
        initialize_ee(project=project)
        if project:
            typer.echo(f"Google Earth Engine initialized successfully with project: {project}")
        else:
            typer.echo("Google Earth Engine initialized successfully with default credentials")
    except Exception as e:
        typer.echo(f"Error initializing Earth Engine: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def embedding(
    lat: Annotated[float, typer.Option(help="Latitude coordinate")],
    lon: Annotated[float, typer.Option(help="Longitude coordinate")], 
    year: Annotated[int, typer.Option(help="Year for the embedding")] = 2024,
    project: Annotated[Optional[str], typer.Option(help="Google Cloud project ID (optional)")] = None,
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
    output: Annotated[Path, typer.Option(help="Output file for results")] = Path("data/samples_with_embeddings.tsv"),
    max_samples: Annotated[int, typer.Option(help="Maximum number of samples to process (for testing)")] = None,
    project: Annotated[str, typer.Option(help="Google Cloud project ID")] = "env-embeddings-2025",
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
    tsv_file: Annotated[Path, typer.Argument(help="Path to TSV file to add embeddings to")],
    output: Annotated[Path, typer.Option(help="Output file with embeddings added")] = None,
    max_rows: Annotated[int, typer.Option(help="Maximum number of rows to process (for testing)")] = None,
    project: Annotated[str, typer.Option(help="Google Cloud project ID")] = "env-embeddings-2025",
    fallback_year: Annotated[int, typer.Option(help="Year to use when original year has no data")] = 2020,
    skip_existing: Annotated[bool, typer.Option(help="Skip rows that already have embeddings")] = True,
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
            skip_existing=skip_existing
        )
        
        typer.echo(f"Successfully added embeddings to {success_count} rows")
        typer.echo(f"Results saved to {output}")
        
    except Exception as e:
        typer.echo(f"Error adding embeddings: {e}", err=True)
        raise typer.Exit(1)

def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
