"""CLI interface for env-embeddings."""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer
from typing_extensions import Annotated

from env_embeddings.earth_engine import initialize_ee, get_embedding

app = typer.Typer(help="env-embeddings: Simple experiment to compare ENVO similarity to google embedding cosine similarity ")

from .embeddings import (
    compute_embedding,
    compute_embeddings_batch,
    cosine_similarity,
    get_similarity_matrix,
    find_most_similar,
)

app = typer.Typer(help="env-embeddings: Simple experiment to compare ENVO similarity to google embedding cosine similarity")


@app.command()
def embed(
    text: Annotated[str, typer.Argument(help="Text to embed")],
    model: Annotated[str, typer.Option(help="Model name")] = "sentence-transformers/all-mpnet-base-v2",
    output: Annotated[Optional[Path], typer.Option(help="Output file for embedding")] = None,
):
    """Compute embedding for a single text."""
    embedding = compute_embedding(text, model)

    if output:
        np.save(output, embedding)
        typer.echo(f"Embedding saved to {output}")
    else:
        typer.echo(f"Embedding shape: {embedding.shape}")
        typer.echo(f"First 5 values: {embedding[:5].tolist()}")


@app.command()
def batch_embed(
    input_file: Annotated[Path, typer.Argument(help="File containing texts (one per line)")],
    output: Annotated[Path, typer.Option(help="Output file for embeddings")] = Path("embeddings.npy"),
    model: Annotated[str, typer.Option(help="Model name")] = "sentence-transformers/all-mpnet-base-v2",
):
    """Compute embeddings for multiple texts from a file."""
    if not input_file.exists():
        typer.echo(f"Error: {input_file} not found", err=True)
        raise typer.Exit(1)

    texts = input_file.read_text().strip().split("\n")
    texts = [t.strip() for t in texts if t.strip()]

    typer.echo(f"Computing embeddings for {len(texts)} texts...")
    embeddings = compute_embeddings_batch(texts, model)

    np.save(output, embeddings)
    typer.echo(f"Embeddings saved to {output} with shape {embeddings.shape}")


@app.command()
def similarity(
    text1: Annotated[str, typer.Argument(help="First text")],
    text2: Annotated[str, typer.Argument(help="Second text")],
    model: Annotated[str, typer.Option(help="Model name")] = "sentence-transformers/all-mpnet-base-v2",
):
    """Compute cosine similarity between two texts."""
    embedding1 = compute_embedding(text1, model)
    embedding2 = compute_embedding(text2, model)

    sim = cosine_similarity(embedding1, embedding2)
    typer.echo(f"Cosine similarity: {sim:.4f}")


@app.command()
def similarity_matrix(
    input_file: Annotated[Path, typer.Argument(help="File containing texts (one per line)")],
    output: Annotated[Optional[Path], typer.Option(help="Output file for similarity matrix")] = None,
    model: Annotated[str, typer.Option(help="Model name")] = "sentence-transformers/all-mpnet-base-v2",
):
    """Compute pairwise similarity matrix for texts."""
    if not input_file.exists():
        typer.echo(f"Error: {input_file} not found", err=True)
        raise typer.Exit(1)

    texts = input_file.read_text().strip().split("\n")
    texts = [t.strip() for t in texts if t.strip()]

    typer.echo(f"Computing embeddings for {len(texts)} texts...")
    embeddings = compute_embeddings_batch(texts, model)

    typer.echo("Computing similarity matrix...")
    sim_matrix = get_similarity_matrix(embeddings)

    if output:
        if output.suffix == ".json":
            # Save as JSON with labels
            result = {
                "texts": texts,
                "matrix": sim_matrix.tolist()
            }
            output.write_text(json.dumps(result, indent=2))
        else:
            np.save(output, sim_matrix)
        typer.echo(f"Similarity matrix saved to {output}")
    else:
        # Display matrix
        typer.echo("\nSimilarity Matrix:")
        typer.echo("-" * 50)
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                if i < j:  # Only show upper triangle
                    typer.echo(f"{text1[:30]:30s} <-> {text2[:30]:30s}: {sim_matrix[i,j]:.4f}")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Query text")],
    corpus_file: Annotated[Path, typer.Argument(help="File containing corpus texts")],
    top_k: Annotated[int, typer.Option(help="Number of results")] = 5,
    model: Annotated[str, typer.Option(help="Model name")] = "sentence-transformers/all-mpnet-base-v2",
):
    """Find most similar texts in a corpus."""
    if not corpus_file.exists():
        typer.echo(f"Error: {corpus_file} not found", err=True)
        raise typer.Exit(1)

    corpus = corpus_file.read_text().strip().split("\n")
    corpus = [t.strip() for t in corpus if t.strip()]

    typer.echo(f"Searching {len(corpus)} texts for: '{query}'")
    results = find_most_similar(query, corpus, model, top_k)

    typer.echo(f"\nTop {min(top_k, len(results))} results:")
    typer.echo("-" * 50)
    for i, (text, score) in enumerate(results, 1):
        typer.echo(f"{i}. [{score:.4f}] {text}")



@app.command()
def init_ee(
    project: Annotated[str, typer.Option(help="Google Cloud project ID (optional)")] = None
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
    project: Annotated[str, typer.Option(help="Google Cloud project ID (optional)")] = None,
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

def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
