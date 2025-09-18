"""CLI interface for env-embeddings."""

import typer
from typing_extensions import Annotated

from env_embeddings.earth_engine import initialize_ee, get_embedding

app = typer.Typer(help="env-embeddings: Simple experiment to compare ENVO similarity to google embedding cosine similarity ")


@app.command()
def run(
    name: Annotated[str, typer.Option(help="Name of the person to greet")],
):
    typer.echo(f"Hello, {name}!")


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
