# Google Earth Engine Embeddings

This document shows how to download Google Earth Engine embeddings using the methods from this codebase.

## Self-Contained Code Snippet

```python
import ee
from typing import List, Optional

def initialize_ee(project: Optional[str] = None) -> None:
    """Initialize Google Earth Engine authentication and session."""
    try:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as e:
        raise Exception(f"Earth Engine initialization failed: {e}")

def get_embedding(lat: float, lon: float, year: int, project: Optional[str] = None) -> List[float]:
    """Return the 64-dimensional AlphaEarth embedding for the given lat/lon and year."""
    # Initialize if project provided
    if project:
        initialize_ee(project)

    # Create point geometry
    point = ee.Geometry.Point(lon, lat)

    # Load Google's satellite embedding collection
    collection = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

    # Filter by year and location
    start = f"{year}-01-01"
    end = f"{year+1}-01-01"
    filtered_collection = collection.filterDate(start, end).filterBounds(point)

    # Get the first (and likely only) image for that year
    image_for_year = filtered_collection.first()

    # Sample the point to get embedding vector (bands A00-A63)
    sampled = image_for_year.sample(region=point, scale=10).first()
    band_dict = sampled.toDictionary().getInfo()

    # Extract 64-dimensional embedding vector
    embedding = [band_dict.get(f"A{str(i).zfill(2)}") for i in range(64)]

    return embedding

# Usage example:
# initialize_ee("your-project-id")
# embedding = get_embedding(39.0372, -121.8036, 2024)
# print(f"64-dimensional vector: {embedding}")
```

## Key Points

- Uses Google's `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` dataset
- Returns 64-dimensional AlphaEarth embeddings (bands A00-A63)
- Requires Earth Engine authentication and project setup
- Works with years 2017-2024 for best coverage

## CLI Usage

The codebase also provides a CLI interface:

```bash
# Get embedding for specific coordinates and year
uv run env-embeddings embedding --lat 39.0372 --lon -121.8036 --year 2024 --project env-embeddings-2025
```

This returns a 64-dimensional embedding vector representing satellite imagery features for that location and time.

## Setup Requirements

1. Google Cloud project with Earth Engine API enabled
2. Earth Engine authentication (`ee.Authenticate()`)
3. Project ID for initialization
4. Python package: `earthengine-api>=1.6.8`
