"""Google Earth Engine embedding utilities."""

import ee
from typing import List


def initialize_ee(project: str = None) -> None:
    """Initialize Google Earth Engine authentication and session.
    
    This will prompt you to authenticate if not already done.
    
    Args:
        project: Google Cloud project ID to use. If None, uses default credentials.
        
    Raises:
        Exception: If Earth Engine initialization fails. Common causes:
                  - Earth Engine API not enabled for the project
                  - Missing serviceusage.services.use permission
                  - No valid project specified
    """
    try:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as e:
        if "no project found" in str(e).lower():
            raise Exception(
                "No Google Cloud project specified. You need to:\n"
                "1. Create a Google Cloud project\n"
                "2. Enable the Earth Engine API for that project\n"
                "3. Grant your account the necessary permissions\n"
                "4. Specify the project: --project YOUR_PROJECT_ID\n"
                f"Original error: {e}"
            )
        elif "required permission" in str(e).lower():
            raise Exception(
                "Missing permissions for Earth Engine. You need to:\n"
                "1. Enable the Earth Engine API for your project\n"
                "2. Grant your account the 'roles/serviceusage.serviceUsageConsumer' role\n"
                "3. Wait a few minutes for permissions to propagate\n"
                f"Original error: {e}"
            )
        else:
            raise e


def get_embedding(lat: float, lon: float, year: int, project: str = None) -> List[float]:
    """Return the 64-dimensional AlphaEarth embedding for the given lat/lon and year.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate  
        year: Year for which to get the embedding
        project: Google Cloud project ID. If provided, will initialize Earth Engine.
        
    Returns:
        List of 64 float values representing the embedding vector
        
    Raises:
        ValueError: If no embedding found for the given coordinates and year
        Exception: If Earth Engine is not initialized and project is not provided
        
    Examples:
        >>> # Note: These doctests require Earth Engine authentication
        >>> # initialize_ee()  # Uncomment to run
        >>> # embedding = get_embedding(39.0372, -121.8036, 2024)
        >>> # len(embedding)
        >>> # 64
    """
    # Try to initialize if project is provided
    if project:
        try:
            initialize_ee(project)
        except Exception:
            pass  # Will fail later with more specific error
    
    # Check if Earth Engine is initialized by trying a simple operation
    try:
        ee.data.getInfo(ee.Number(1))
    except Exception as e:
        if "not initialized" in str(e).lower():
            raise Exception(
                "Earth Engine not initialized. Run 'env-embeddings init-ee --project YOUR_PROJECT_ID' first, "
                "or provide a project parameter."
            )
        # If it's a different error, continue (might be network issue, etc.)
    # Create a point geometry
    point = ee.Geometry.Point(lon, lat)

    # Load the embeddings image collection
    collection = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

    # Filter to the year you want
    # e.g., from Jan 1 of the given year to Jan 1 of year+1
    start = f"{year}-01-01"
    end = f"{year+1}-01-01"

    image_for_year = (collection
                      .filterDate(start, end)
                      .filterBounds(point)
                      .first())

    if image_for_year is None:
        raise ValueError(f"No embedding found for {lat},{lon} at year {year}")

    # Sample the point to get the embedding vector  
    # 'sample' returns a FeatureCollection with properties A00..A63
    # scale parameter should match resolution (10 m)
    sampled = image_for_year.sample(region=point, scale=10).first()

    # Get the dictionary of band values
    band_dict = sampled.toDictionary().getInfo()

    # Extract the embedding vector
    emb = [band_dict.get(f"A{str(i).zfill(2)}") for i in range(64)]

    return emb


if __name__ == "__main__":
    initialize_ee()
    lat = 39.0372
    lon = -121.8036
    year = 2024
    vec = get_embedding(lat, lon, year)
    print("Embedding vector (len={}):".format(len(vec)))
    print(vec)