"""Google Earth Engine embedding utilities."""

from pathlib import Path
from typing import List, Optional
import time

import ee
from diskcache import Cache

# Disk cache using diskcache library
_CACHE_DIR = Path.home() / ".cache" / "env-embeddings" / "google_earth"
_cache = Cache(str(_CACHE_DIR))


def initialize_ee(project: Optional[str] = None) -> None:
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


def _retry_with_exponential_backoff(
    func, max_retries: int = 5, initial_delay: float = 1.0
):
    """Retry a function with exponential backoff for rate limiting.

    Args:
        func: Function to retry (should be a lambda/callable with no arguments)
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (doubles with each retry)

    Returns:
        Result of the function call

    Raises:
        Exception: Re-raises the last exception if all retries are exhausted
    """
    delay = initial_delay
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()

            # Check if it's a rate limit error (HTTP 429)
            if (
                "429" in error_msg
                or "too many requests" in error_msg
                or "quota" in error_msg
            ):
                if attempt < max_retries - 1:
                    print(
                        f"Rate limit hit (429), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue

            # For non-rate-limit errors, raise immediately
            raise

    # If we've exhausted all retries, raise the last exception
    if last_exception is not None:
        raise last_exception
    else:
        # This should never happen, but satisfy mypy
        raise RuntimeError("Retry logic failed without capturing an exception")


def get_embedding(
    lat: float,
    lon: float,
    year: int,
    project: Optional[str] = None,
    use_cache: bool = True,
) -> List[float]:
    """Return the 64-dimensional AlphaEarth embedding for the given lat/lon and year with disk-backed caching.

    Uses diskcache for automatic persistent caching to SQLite.

    Coordinates are normalized to 4 decimal places for cache key consistency,
    avoiding duplicate cache entries for semantically identical coordinates
    with different representations (e.g., 35.1180 vs 35.118).

    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        year: Year for which to get the embedding
        project: Google Cloud project ID. If provided, will initialize Earth Engine.
        use_cache: Whether to use cached embeddings (default: True)

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
    # Normalize coordinates to 4 decimal places for cache key consistency
    # This ensures that 35.1180 and 35.118 produce the same cache key
    lat_normalized = round(float(lat), 4)
    lon_normalized = round(float(lon), 4)

    # Create cache key with normalized coordinates
    cache_key = (lat_normalized, lon_normalized, year)

    # Check cache first (including cached failures)
    if use_cache and cache_key in _cache:
        cached_value = _cache[cache_key]
        # If cached value is None, it means we previously failed for this location/year
        if cached_value is None:
            raise ValueError(
                f"No embedding found for {lat},{lon} at year {year} (cached failure)"
            )
        return cached_value

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
    # Create a point geometry using normalized coordinates
    point = ee.Geometry.Point(lon_normalized, lat_normalized)

    # Load the embeddings image collection
    collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    # Filter to the year you want
    # e.g., from Jan 1 of the given year to Jan 1 of year+1
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    filtered_collection = collection.filterDate(start, end).filterBounds(point)

    # Wrap all Earth Engine API calls with retry logic for rate limiting
    def _fetch_embedding():
        # Check if collection has any images before calling first()
        size = filtered_collection.size()
        collection_size = size.getInfo()
        if collection_size == 0:
            raise ValueError(
                f"No embedding found for {lat},{lon} at year {year}. The Google Satellite Embedding dataset may not have coverage for this location/year combination. Try years 2017-2024 for better coverage."
            )

        image_for_year = filtered_collection.first()
        if image_for_year is None:
            raise ValueError(
                f"No embedding found for {lat},{lon} at year {year}. The Google Satellite Embedding dataset may not have coverage for this location/year combination. Try years 2017-2024 for better coverage."
            )

        # Sample the point to get the embedding vector
        # 'sample' returns a FeatureCollection with properties A00..A63
        # scale parameter should match resolution (10 m)
        sampled = image_for_year.sample(region=point, scale=10).first()

        # Get the dictionary of band values
        band_dict = sampled.toDictionary().getInfo()

        # Extract the embedding vector
        return [band_dict.get(f"A{str(i).zfill(2)}") for i in range(64)]

    # Execute with retry logic, catching failures to cache them
    try:
        emb = _retry_with_exponential_backoff(_fetch_embedding)

        # Cache the successful result
        if use_cache:
            _cache[cache_key] = emb

        return emb
    except ValueError as e:
        # Cache the failure as None for fast-fail on subsequent attempts
        if use_cache and "No embedding found" in str(e):
            _cache[cache_key] = None
        raise


def get_cache_stats() -> dict:
    """Get statistics about the Google Earth embedding cache.

    Returns:
        Dictionary with cache statistics
    """
    return {"total_cached": len(_cache), "cache_dir": str(_CACHE_DIR)}


def clear_cache() -> None:
    """Clear the Google Earth embedding cache."""
    _cache.clear()
    print(f"Cleared Google Earth cache at {_CACHE_DIR}")


if __name__ == "__main__":
    initialize_ee()
    lat = 39.0372
    lon = -121.8036
    year = 2024
    vec = get_embedding(lat, lon, year)
    print("Embedding vector (len={}):".format(len(vec)))
    print(vec)
