"""Process sample data and retrieve Earth Engine embeddings."""

import re
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from .earth_engine import (
    get_embedding,
    initialize_ee,
    get_cache_stats as get_ge_cache_stats,
    _cache as ge_cache,
)
import ee
from .envo_embeddings import (
    get_envo_embedding_from_text,
    extract_first_envo_term,
    get_envo_embedding,
    get_cache_stats,
)


@dataclass
class ProcessingStats:
    """Track detailed statistics during embedding processing."""

    cache_hits: int = 0
    cache_misses: int = 0
    api_success: int = 0
    api_failures_no_coverage: int = 0
    api_failures_rate_limit: int = 0
    api_failures_other: int = 0
    rows_skipped_existing: int = 0
    rows_skipped_invalid_data: int = 0

    def total_processed(self) -> int:
        return self.cache_hits + self.api_success

    def total_failed(self) -> int:
        return (
            self.api_failures_no_coverage
            + self.api_failures_rate_limit
            + self.api_failures_other
        )

    def print_summary(self, title: str = "Processing Summary"):
        """Print a detailed summary of processing statistics."""
        print(f"\n=== {title} ===")
        print(f"Successfully retrieved embeddings: {self.total_processed()}")
        print(f"  - From cache: {self.cache_hits}")
        print(f"  - From API: {self.api_success}")
        print(f"\nFailed to retrieve embeddings: {self.total_failed()}")
        print(f"  - No coverage/ocean: {self.api_failures_no_coverage}")
        print(f"  - Rate limit (429): {self.api_failures_rate_limit}")
        print(f"  - Other errors: {self.api_failures_other}")
        print("\nSkipped rows:")
        print(f"  - Already had embeddings: {self.rows_skipped_existing}")
        print(
            f"  - Invalid data (missing coords/dates): {self.rows_skipped_invalid_data}"
        )


def parse_coordinate_string(coord_str: str) -> Optional[Tuple[float, float]]:
    """Parse coordinate string like '50.936 N 6.952 E' to (lat, lon).

    Args:
        coord_str: Coordinate string in various formats

    Returns:
        Tuple of (latitude, longitude) in decimal degrees, or None if invalid

    Examples:
        >>> parse_coordinate_string("50.936 N 6.952 E")
        (50.936, 6.952)
        >>> parse_coordinate_string("35.32 S 148.25 E")
        (-35.32, 148.25)
        >>> parse_coordinate_string("28.1000 N 81.6000 W")
        (28.1, -81.6)
    """
    if not coord_str or coord_str.strip() == "":
        return None

    try:
        # Pattern for coordinates like "50.936 N 6.952 E"
        pattern = r"([0-9.-]+)\s*([NSEW])\s+([0-9.-]+)\s*([NSEW])"
        match = re.match(pattern, coord_str.strip())

        if not match:
            return None

        lat_val, lat_dir, lon_val, lon_dir = match.groups()

        lat = float(lat_val)
        lon = float(lon_val)

        # Apply direction multipliers
        if lat_dir.upper() == "S":
            lat = -lat
        if lon_dir.upper() == "W":
            lon = -lon

        return (lat, lon)

    except (ValueError, AttributeError):
        return None


def parse_date_to_year(date_str) -> Optional[int]:
    """Parse various date formats to extract year.

    Args:
        date_str: Date string or numeric year in various formats

    Returns:
        Year as integer, or None if invalid

    Examples:
        >>> parse_date_to_year("2008-08-20")
        2008
        >>> parse_date_to_year("2016")
        2016
        >>> parse_date_to_year("1998-12")
        1998
        >>> parse_date_to_year(2020)
        2020
        >>> parse_date_to_year(2020.0)
        2020
    """
    # Handle None/empty
    if date_str is None or (isinstance(date_str, str) and date_str.strip() == ""):
        return None

    # Handle numeric input (int or float)
    if isinstance(date_str, (int, float)):
        # Check for NaN
        import math

        if math.isnan(date_str):
            return None
        year = int(date_str)
        if 1950 <= year <= 2030:
            return year
        return None

    # Handle string input
    try:
        # Extract first 4-digit number (assuming it's the year)
        year_match = re.search(r"\b(19|20)\d{2}\b", str(date_str))
        if year_match:
            year = int(year_match.group())
            # Sanity check - reasonable year range for biological samples
            if 1950 <= year <= 2030:
                return year
        return None

    except (ValueError, AttributeError):
        return None


def load_sample_data(tsv_path: Path, max_samples: Optional[int] = None) -> pd.DataFrame:
    """Load and parse sample data from TSV file.

    Args:
        tsv_path: Path to the TSV file
        max_samples: Maximum number of samples to load (for testing)

    Returns:
        DataFrame with parsed coordinates and dates
    """
    # Read TSV file
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    if max_samples:
        df = df.head(max_samples)

    # Parse coordinates
    coord_data = df["lat_lon"].apply(parse_coordinate_string)
    df["parsed_lat"] = coord_data.apply(lambda x: x[0] if x else None)
    df["parsed_lon"] = coord_data.apply(lambda x: x[1] if x else None)

    # Parse dates to years
    df["parsed_year"] = df["date"].apply(parse_date_to_year)

    # Filter out samples with missing critical data
    valid_samples = df.dropna(subset=["parsed_lat", "parsed_lon", "parsed_year"])

    print(
        f"Loaded {len(df)} samples, {len(valid_samples)} have valid coordinates and dates"
    )

    return valid_samples


def get_sample_embedding(
    sample: pd.Series, project: Optional[str] = None, fallback_year: int = 2020
) -> Optional[List[float]]:
    """Get Earth Engine embedding for a single sample.

    Args:
        sample: Pandas Series with parsed_lat, parsed_lon, parsed_year
        project: Google Cloud project ID
        fallback_year: Year to use if original year fails (default 2020)

    Returns:
        64-dimensional embedding vector or None if failed
    """
    try:
        lat = sample["parsed_lat"]
        lon = sample["parsed_lon"]
        year = int(sample["parsed_year"])

        # Ensure project is provided for embedding function
        if not project:
            project = "env-embeddings-2025"

        # Try original year first
        try:
            embedding = get_embedding(lat, lon, year, project)
            return embedding
        except ValueError as e:
            if "No embedding found" in str(e):
                # Try fallback year (Google Satellite Embeddings may not have old data)
                print(f"No embedding for {lat},{lon} in {year}, trying {fallback_year}")
                embedding = get_embedding(lat, lon, fallback_year, project)
                return embedding
            else:
                raise e

    except Exception as e:
        print(
            f"Error getting embedding for sample {sample.get('genome_id', 'unknown')}: {e}"
        )
        return None


def process_samples_batch(
    samples: pd.DataFrame,
    project: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Process multiple samples to get Earth Engine embeddings.

    Args:
        samples: DataFrame with sample data
        project: Google Cloud project ID
        output_path: Optional path to save results

    Returns:
        DataFrame with embeddings added
    """
    # Initialize Earth Engine if not already done
    if not project:
        project = "env-embeddings-2025"

    try:
        print(f"Initializing Earth Engine with project: {project}")
        initialize_ee(project)
        print("Earth Engine initialized successfully")
    except Exception as e:
        print(f"Warning: Earth Engine initialization failed: {e}")
        print("Proceeding anyway - Earth Engine may already be initialized")

    print(f"Processing {len(samples)} samples for Earth Engine embeddings...")

    embeddings = []
    successful_count = 0

    for idx, sample in samples.iterrows():
        embedding = get_sample_embedding(sample, project)
        embeddings.append(embedding)

        if embedding is not None:
            successful_count += 1

        # Progress update every 100 samples
        row_num = int(idx) + 1 if isinstance(idx, int) else 0
        if row_num % 100 == 0:
            print(
                f"Processed {row_num}/{len(samples)} samples, {successful_count} successful"
            )

    # Add embeddings to dataframe
    samples_with_embeddings = samples.copy()
    samples_with_embeddings["earth_engine_embedding"] = embeddings

    # Filter to only successful embeddings
    successful_samples = samples_with_embeddings.dropna(
        subset=["earth_engine_embedding"]
    )

    print(
        f"Successfully retrieved embeddings for {len(successful_samples)}/{len(samples)} samples"
    )

    if output_path:
        successful_samples.to_csv(output_path, sep="\t", index=False)
        print(f"Results saved to {output_path}")

    return successful_samples


def add_embeddings_to_tsv(
    tsv_file: Path,
    output_file: Path,
    max_rows: Optional[int] = None,
    project: str = "env-embeddings-2025",
    fallback_year: int = 2020,
    skip_existing: bool = True,
) -> int:
    """Add Google Earth Engine embeddings to an existing TSV file.

    Args:
        tsv_file: Path to input TSV file
        output_file: Path to output TSV file
        max_rows: Maximum number of rows to process (for testing)
        project: Google Cloud project ID
        fallback_year: Year to use when original year has no satellite data
        skip_existing: Skip rows that already have embeddings

    Returns:
        Number of rows successfully processed with embeddings
    """
    # Initialize Earth Engine
    if not project:
        project = "env-embeddings-2025"

    try:
        print(f"Initializing Earth Engine with project: {project}")
        initialize_ee(project)
        print("Earth Engine initialized successfully")
    except Exception as e:
        print(f"Warning: Earth Engine initialization failed: {e}")
        print("Proceeding anyway - Earth Engine may already be initialized")

    # Read the TSV file
    print(f"Loading TSV file: {tsv_file}")
    df = pd.read_csv(tsv_file, sep="\t", low_memory=False)

    if max_rows:
        df = df.head(max_rows)
        print(f"Processing first {max_rows} rows for testing")

    # Add google_earth_embeddings column if it doesn't exist
    if "google_earth_embeddings" not in df.columns:
        df["google_earth_embeddings"] = None
        print("Added 'google_earth_embeddings' column")

    # Add envo_embeddings column if it doesn't exist
    if "envo_embeddings" not in df.columns:
        df["envo_embeddings"] = None
        print("Added 'envo_embeddings' column")

    success_count = 0
    total_rows = len(df)

    print(f"Processing {total_rows} rows...")

    # Create progress bar
    with tqdm(total=total_rows, desc="Processing samples", unit="row") as pbar:
        for idx, row in df.iterrows():
            row_num: int = int(idx) if isinstance(idx, int) else 0
            # Skip if already has embeddings and skip_existing is True
            if skip_existing and pd.notna(row.get("google_earth_embeddings")):
                pbar.update(1)
                continue

            # Parse coordinates from lat_lon column
            lat_lon_str = row.get("lat_lon", "")
            coords = parse_coordinate_string(lat_lon_str)

            if coords is None:
                tqdm.write(f"Row {row_num + 1}: Invalid coordinates '{lat_lon_str}'")
                pbar.update(1)
                continue

            lat, lon = coords

            # Parse year from date column
            date_str = row.get("date", "")
            year = parse_date_to_year(date_str)

            if year is None:
                tqdm.write(f"Row {row_num + 1}: Invalid date '{date_str}'")
                pbar.update(1)
                continue

            # Get Google Earth Engine embedding
            try:
                # Try original year first
                try:
                    embedding = get_embedding(lat, lon, year, project)
                    df.at[idx, "google_earth_embeddings"] = str(embedding)  # type: ignore[index]
                    success_count += 1
                    tqdm.write(
                        f"Row {row_num + 1}: Got embedding for {lat},{lon} in {year}"
                    )

                except ValueError as e:
                    if "No embedding found" in str(e):
                        # Try fallback year
                        tqdm.write(
                            f"Row {row_num + 1}: No embedding for {lat},{lon} in {year}, trying {fallback_year}"
                        )
                        embedding = get_embedding(lat, lon, fallback_year, project)
                        df.at[idx, "google_earth_embeddings"] = str(embedding)  # type: ignore[index]
                        success_count += 1
                        tqdm.write(
                            f"Row {row_num + 1}: Got embedding for {lat},{lon} in {fallback_year} (fallback)"
                        )
                    else:
                        tqdm.write(
                            f"Row {row_num + 1}: Error with coordinates {lat},{lon}: {e}"
                        )

            except Exception as e:
                tqdm.write(
                    f"Row {row_num + 1}: Failed to get embedding for {lat},{lon}: {e}"
                )

            # Get ENVO embedding if env_broad_med_local column exists and has data
            envo_text = row.get("env_broad_med_local", "")
            if envo_text and isinstance(envo_text, str) and envo_text.strip():
                try:
                    envo_embedding = get_envo_embedding_from_text(envo_text)
                    if envo_embedding:
                        df.at[idx, "envo_embeddings"] = str(envo_embedding)  # type: ignore[index]
                        tqdm.write(
                            f"Row {row_num + 1}: Got ENVO embedding from '{envo_text[:50]}...'"
                        )
                    else:
                        tqdm.write(
                            f"Row {row_num + 1}: No valid ENVO term found in '{envo_text[:50]}...'"
                        )
                except Exception as e:
                    tqdm.write(f"Row {row_num + 1}: Failed to get ENVO embedding: {e}")

            # Update progress bar
            pbar.set_postfix({"Success": f"{success_count}/{row_num + 1}"})
            pbar.update(1)

    # Save the updated TSV
    print(f"Saving results to {output_file}")
    df.to_csv(output_file, sep="\t", index=False)

    print(
        f"Completed! Successfully added embeddings to {success_count}/{total_rows} rows"
    )
    return success_count


def add_envo_embeddings_to_tsv(
    tsv_file: Path,
    output_file: Path,
    max_rows: Optional[int] = None,
    skip_existing: bool = True,
) -> int:
    """Add ENVO embeddings to an existing TSV file.

    Args:
        tsv_file: Path to input TSV file
        output_file: Path to output TSV file
        max_rows: Maximum number of rows to process (for testing)
        skip_existing: Skip rows that already have ENVO embeddings

    Returns:
        Number of rows successfully processed with ENVO embeddings
    """
    # Read the TSV file
    print(f"Loading TSV file: {tsv_file}")
    df = pd.read_csv(tsv_file, sep="\t", low_memory=False)

    if max_rows:
        df = df.head(max_rows)
        print(f"Processing first {max_rows} rows for testing")

    # Add envo_embeddings column if it doesn't exist
    if "envo_embeddings" not in df.columns:
        df["envo_embeddings"] = None
        print("Added 'envo_embeddings' column")

    success_count = 0
    total_rows = len(df)

    print(f"Processing {total_rows} rows for ENVO embeddings...")

    # Create progress bar
    with tqdm(total=total_rows, desc="Processing ENVO embeddings", unit="row") as pbar:
        for idx, row in df.iterrows():
            row_num: int = int(idx) if isinstance(idx, int) else 0
            # Skip if already has ENVO embeddings and skip_existing is True
            if skip_existing and pd.notna(row.get("envo_embeddings")):
                pbar.update(1)
                continue

            # Get ENVO embedding if env_broad_med_local column exists and has data
            envo_text = row.get("env_broad_med_local", "")
            if envo_text and isinstance(envo_text, str) and envo_text.strip():
                try:
                    envo_embedding = get_envo_embedding_from_text(envo_text)
                    if envo_embedding:
                        df.at[idx, "envo_embeddings"] = str(envo_embedding)  # type: ignore[index]
                        success_count += 1
                        tqdm.write(
                            f"Row {row_num + 1}: Got ENVO embedding from '{envo_text[:50]}...'"
                        )
                    else:
                        tqdm.write(
                            f"Row {row_num + 1}: No valid ENVO term found in '{envo_text[:50]}...'"
                        )
                except Exception as e:
                    tqdm.write(f"Row {row_num + 1}: Failed to get ENVO embedding: {e}")
            else:
                tqdm.write(
                    f"Row {row_num + 1}: No ENVO text found in env_broad_med_local column"
                )

            # Update progress bar
            pbar.set_postfix({"Success": f"{success_count}/{row_num + 1}"})
            pbar.update(1)

    # Save the updated TSV
    print(f"Saving results to {output_file}")
    df.to_csv(output_file, sep="\t", index=False)

    print(
        f"Completed! Successfully added ENVO embeddings to {success_count}/{total_rows} rows"
    )
    return success_count


def normalize_biosamples_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw biosample coordinates and dates to normalized format.

    Transforms:
        collection_date_raw → collection_date (cleaned string, preserves full date)
        lat_lon_raw → latitude, longitude (numeric)

    Year extraction happens later when fetching Google Earth embeddings.
    Efficiently deduplicates parsing - processes each unique value only once.

    Args:
        df: DataFrame with collection_date_raw and lat_lon_raw columns

    Returns:
        DataFrame with added latitude, longitude, collection_date columns
    """
    # Parse unique coordinates once (much faster for datasets with duplicates)
    unique_coords = df["lat_lon_raw"].dropna().unique()
    coord_map = {coord: parse_coordinate_string(str(coord)) for coord in unique_coords}

    # Add None mapping for missing values
    coord_map[None] = None
    if pd.isna(None):
        coord_map[pd.NA] = None

    # Map back to dataframe
    parsed_coords = df["lat_lon_raw"].map(coord_map)
    df["latitude"] = parsed_coords.apply(lambda x: x[0] if x else None)
    df["longitude"] = parsed_coords.apply(lambda x: x[1] if x else None)

    # Normalize dates - keep full date string, just clean it
    # Year extraction happens downstream when fetching embeddings
    unique_dates = df["collection_date_raw"].dropna().unique()
    # Just clean/strip whitespace, keep the full date
    date_map: dict = {date: str(date).strip() for date in unique_dates}

    # Add None mapping for missing values
    date_map[None] = None  # type: ignore[index]
    if pd.isna(None):
        date_map[pd.NA] = None  # type: ignore[index]

    df["collection_date"] = df["collection_date_raw"].map(date_map)

    return df


def normalize_biosamples(
    input_file: Path,
    output_file: Path,
) -> int:
    """Normalize biosample data - parse raw coordinates and dates without fetching embeddings.

    Converts raw format to normalized format:
        Input:  collection_date_raw, lat_lon_raw
        Output: collection_date (year), latitude, longitude (numeric)

    Useful for:
        - Pre-processing before batch embedding retrieval
        - Inspecting parsed data before API calls
        - Debugging coordinate/date parsing issues

    Args:
        input_file: Path to input CSV or TSV file with raw format
        output_file: Path to output file (same format as input)

    Returns:
        Number of rows successfully normalized
    """
    # Auto-detect file format (CSV vs TSV)
    delimiter = "\t" if input_file.suffix.lower() == ".tsv" else ","
    file_type = "TSV" if delimiter == "\t" else "CSV"

    # Read the file
    print(f"Loading {file_type} file: {input_file}")
    df = pd.read_csv(input_file, sep=delimiter, low_memory=False)

    # Check for required columns
    if "collection_date_raw" not in df.columns or "lat_lon_raw" not in df.columns:
        raise ValueError(
            f"Input file must have 'collection_date_raw' and 'lat_lon_raw' columns.\n"
            f"Found columns: {list(df.columns)}"
        )

    print(f"Loaded {len(df)} rows")
    print("\nNormalizing coordinates and dates...")

    # Parse using shared function
    print("  - Parsing lat_lon_raw → latitude, longitude...")
    print(
        "  - Normalizing collection_date_raw → collection_date (full date preserved)..."
    )
    df = normalize_biosamples_dataframe(df)

    parsed_coords = df["latitude"].notna().sum()
    normalized_dates = df["collection_date"].notna().sum()

    print(
        f"    ✓ Parsed {parsed_coords}/{len(df)} coordinates ({parsed_coords / len(df) * 100:.1f}%)"
    )
    print(
        f"    ✓ Normalized {normalized_dates}/{len(df)} dates ({normalized_dates / len(df) * 100:.1f}%)"
    )

    # Count rows with both parsed
    both_parsed = (
        (df["latitude"].notna())
        & (df["longitude"].notna())
        & (df["collection_date"].notna())
    ).sum()
    print(
        f"\n✓ {both_parsed}/{len(df)} rows have complete normalized data ({both_parsed / len(df) * 100:.1f}%)"
    )

    # Save normalized file
    print(f"\nSaving normalized data to {output_file}")
    df.to_csv(output_file, sep=delimiter, index=False)

    print("✓ Done!")
    return both_parsed


def get_embeddings_batch(
    coords_batch: List[Tuple[float, float, int]],
    project: Optional[str] = None,
) -> dict:
    """Fetch Google Earth embeddings for a batch of coordinates using sampleRegions.

    Args:
        coords_batch: List of (latitude, longitude, year) tuples
        project: Google Cloud project ID

    Returns:
        Dictionary mapping (lat, lon, year) -> embedding list

    Raises:
        Exception: If Earth Engine API call fails
    """
    if not coords_batch:
        return {}

    # Group by year since we need different images for different years
    by_year: dict = {}
    for lat, lon, year in coords_batch:
        if year not in by_year:
            by_year[year] = []
        by_year[year].append((lat, lon))

    results = {}

    for year, coords in by_year.items():
        # Process each coordinate - can't batch sampleRegions across different tiles
        # but we can batch the getInfo() calls by building all the sampling operations
        # and retrieving results together
        for lat, lon in coords:
            try:
                start = ee.Date.fromYMD(year, 1, 1)
                end = start.advance(1, "year")

                point = ee.Geometry.Point([lon, lat])

                # Filter to images that contain this point
                image = (
                    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
                    .filterDate(start, end)
                    .filterBounds(point)
                    .first()
                )

                if image is None:
                    continue  # No coverage

                # Sample the point
                sampled = image.sample(region=point, scale=10).first()

                # Get embedding
                band_dict = sampled.toDictionary().getInfo()

                # Extract A00...A63
                embedding = [band_dict.get(f"A{str(i).zfill(2)}") for i in range(64)]

                # Check if we got valid data
                if all(v is not None for v in embedding):
                    results[(lat, lon, year)] = embedding
            except Exception:
                # Skip this coordinate on any error
                continue

    return results


def add_google_earth_embeddings_to_csv(
    csv_file: Path,
    output_file: Path,
    max_rows: Optional[int] = None,
    project: str = "env-embeddings-2025",
    fallback_year: int = 2020,
    skip_existing: bool = True,
    random_sample: bool = True,
) -> int:
    """Legacy function - calls add_google_earth_embeddings_flexible().

    Maintained for backwards compatibility. New code should use
    add_google_earth_embeddings_flexible() which handles both CSV/TSV
    and raw/normalized formats.
    """
    return add_google_earth_embeddings_flexible(
        input_file=csv_file,
        output_file=output_file,
        max_rows=max_rows,
        project=project,
        fallback_year=fallback_year,
        skip_existing=skip_existing,
        random_sample=random_sample,
    )


def add_google_earth_embeddings_flexible(
    input_file: Path,
    output_file: Path,
    max_rows: Optional[int] = None,
    project: str = "env-embeddings-2025",
    fallback_year: int = 2020,
    skip_existing: bool = True,
    random_sample: bool = True,
    batch_size: Optional[int] = None,
) -> int:
    """Add Google Earth Engine embeddings - handles both raw and normalized formats.

    Auto-detects and handles two input formats:

    Format 1 (Raw - from SQL extract):
        - collection_date_raw, lat_lon_raw (strings to be parsed)
        - Parses coordinates and dates using parse_coordinate_string() and parse_date_to_year()

    Format 2 (Normalized - already processed):
        - collection_date, latitude, longitude (already parsed)
        - Uses values directly

    File type (CSV/TSV) is auto-detected from file extension.

    Rows with missing embeddings are filtered out from the final output.

    Args:
        input_file: Path to input CSV or TSV file
        output_file: Path to output file (same format as input)
        max_rows: Maximum number of rows to process (all rows if None)
        project: Google Cloud project ID
        fallback_year: Year to use when original year has no satellite data
        skip_existing: Skip rows that already have embeddings
        random_sample: If True, randomly sample max_rows (avoids geographic/temporal bias).
                      If False, take first max_rows sequentially (useful for resuming).
                      Default: True (recommended for representative sampling)
        batch_size: If set, process coordinates in batches using Earth Engine's sampleRegions
                   (much faster for large datasets). Recommended: 500 (conservative and safe).
                   If None, processes one-by-one (slower but works with smaller quotas).

    Returns:
        Number of rows successfully processed with embeddings

    Examples:
        >>> # Raw format (from SQL extract)
        >>> add_google_earth_embeddings_flexible(
        ...     Path("data/ncbi_raw.tsv"),  # Has collection_date_raw, lat_lon_raw
        ...     Path("data/ncbi_with_embeddings.tsv")
        ... )
        >>> # Normalized format
        >>> add_google_earth_embeddings_flexible(
        ...     Path("data/normalized.csv"),  # Has collection_date, latitude, longitude
        ...     Path("data/normalized_with_embeddings.csv")
        ... )
    """
    # Initialize Earth Engine
    if not project:
        project = "env-embeddings-2025"

    try:
        print(f"Initializing Earth Engine with project: {project}")
        initialize_ee(project)
        print("Earth Engine initialized successfully")
    except Exception as e:
        print(f"Warning: Earth Engine initialization failed: {e}")
        print("Proceeding anyway - Earth Engine may already be initialized")

    # Auto-detect file format (CSV vs TSV)
    delimiter = "\t" if input_file.suffix.lower() == ".tsv" else ","
    file_type = "TSV" if delimiter == "\t" else "CSV"

    # Read the file
    print(f"Loading {file_type} file: {input_file}")
    df = pd.read_csv(input_file, sep=delimiter, low_memory=False)

    # Detect format and normalize if needed
    has_raw_format = "collection_date_raw" in df.columns and "lat_lon_raw" in df.columns
    has_normalized_format = (
        "collection_date" in df.columns
        and "latitude" in df.columns
        and "longitude" in df.columns
    )

    if has_raw_format:
        print(
            "Detected RAW format (collection_date_raw, lat_lon_raw) - parsing coordinates and normalizing dates..."
        )

        # Use shared normalization function
        df = normalize_biosamples_dataframe(df)

        print(f"  - Parsed {df['latitude'].notna().sum()} lat/lon coordinates")
        print(
            f"  - Normalized {df['collection_date'].notna().sum()} dates (full date preserved, year extracted on-demand)"
        )

    elif has_normalized_format:
        print(
            "Detected NORMALIZED format (collection_date, latitude, longitude) - using directly"
        )
    else:
        raise ValueError(
            "Input file must have either:\n"
            "  - Raw format: collection_date_raw, lat_lon_raw, env_broad_scale, env_local_scale, env_medium\n"
            "  - Normalized format: collection_date, latitude, longitude, env_broad_scale, env_local_scale, env_medium\n"
            f"Found columns: {list(df.columns)}"
        )

    if max_rows:
        if random_sample:
            # Random sample - better for representative sampling
            df = df.sample(n=min(max_rows, len(df)), random_state=42)
            print(
                f"Processing random sample of {len(df)} rows (seed=42 for reproducibility)"
            )
        else:
            # Sequential - first N rows
            df = df.head(max_rows)
            print(f"Processing first {len(df)} rows sequentially")

    # Add google_earth_embeddings column if it doesn't exist
    if "google_earth_embeddings" not in df.columns:
        df["google_earth_embeddings"] = None
        print("Added 'google_earth_embeddings' column")

    # Track detailed statistics
    stats = ProcessingStats()
    total_rows = len(df)

    # Note: Cache hit tracking is done per-request by comparing cache size before/after

    print(f"Processing {total_rows} rows...")

    # BATCHED PROCESSING PATH
    if batch_size and batch_size > 1:
        print(f"Using BATCHED processing (batch_size={batch_size})")
        print("  1. Pre-processing: identifying unique coordinates to fetch...")

        # Collect all (lat, lon, year) that need embeddings
        coords_to_fetch = []
        row_to_coords = {}  # Map row index to (lat, lon, year)

        for idx, row in df.iterrows():
            # Skip if already has embeddings
            if skip_existing and pd.notna(row.get("google_earth_embeddings")):
                stats.rows_skipped_existing += 1
                continue

            lat = row.get("latitude")
            lon = row.get("longitude")

            if pd.isna(lat) or pd.isna(lon):
                stats.rows_skipped_invalid_data += 1
                continue

            date_str = row.get("collection_date", "")
            year = parse_date_to_year(str(date_str))

            if year is None:
                stats.rows_skipped_invalid_data += 1
                continue

            # Normalize coordinates (same as cache key)
            lat_norm = round(float(lat), 4)
            lon_norm = round(float(lon), 4)

            row_to_coords[idx] = (lat_norm, lon_norm, year)
            coords_to_fetch.append((lat_norm, lon_norm, year))

        # Uniquify coordinates
        unique_coords = list(set(coords_to_fetch))
        print(f"  - Found {len(coords_to_fetch)} rows needing embeddings")
        print(
            f"  - {len(unique_coords)} unique coordinates ({len(coords_to_fetch) - len(unique_coords)} duplicates)"
        )

        # Check cache for unique coords
        print("  2. Checking cache...")
        uncached_coords = []
        cached_results = {}

        for coord in unique_coords:
            cache_key = coord  # (lat, lon, year)
            if cache_key in ge_cache:
                cached_val = ge_cache[cache_key]
                if cached_val is not None:
                    cached_results[coord] = cached_val
                    stats.cache_hits += 1
                else:
                    # Cached failure
                    stats.cache_hits += 1
                    stats.api_failures_no_coverage += 1
            else:
                uncached_coords.append(coord)
                stats.cache_misses += 1

        print(
            f"  - {len(cached_results)} in cache, {len(uncached_coords)} need fetching"
        )

        # Batch fetch uncached coords
        if uncached_coords:
            print(
                f"  3. Fetching {len(uncached_coords)} embeddings in batches of {batch_size}..."
            )

            batch_results = {}

            with tqdm(
                total=len(uncached_coords), desc="Fetching batches", unit="coord"
            ) as pbar:
                for i in range(0, len(uncached_coords), batch_size):
                    batch = uncached_coords[i : i + batch_size]

                    try:
                        results = get_embeddings_batch(batch, project)
                        batch_results.update(results)
                        stats.api_success += len(results)

                        # Update cache
                        for coord, embedding in results.items():
                            ge_cache[coord] = embedding

                        # Mark coords that failed (no data returned)
                        for coord in batch:
                            if coord not in results:
                                ge_cache[coord] = None  # Cache the failure
                                stats.api_failures_no_coverage += 1

                        pbar.update(len(batch))
                    except Exception as e:
                        print(f"\nError fetching batch: {e}")
                        stats.api_failures_other += len(batch)
                        pbar.update(len(batch))

            cached_results.update(batch_results)

        # Map results back to dataframe
        print("  4. Mapping results back to dataframe...")
        for idx, coord in row_to_coords.items():
            if coord in cached_results:
                df.at[idx, "google_earth_embeddings"] = str(cached_results[coord])  # type: ignore[index]

        print("  ✓ Batched processing complete!")

    # ONE-BY-ONE PROCESSING PATH (original)
    else:
        if batch_size:
            print(
                f"Note: batch_size={batch_size} is too small, using one-by-one processing"
            )

        # Create progress bar
        with tqdm(
            total=total_rows, desc="Adding Google Earth embeddings", unit="row"
        ) as pbar:
            for idx, row in df.iterrows():
                # Skip if already has embeddings and skip_existing is True
                if skip_existing and pd.notna(row.get("google_earth_embeddings")):
                    stats.rows_skipped_existing += 1
                    pbar.update(1)
                    continue

                # Get latitude and longitude (already numeric)
                lat = row.get("latitude")
                lon = row.get("longitude")

                if pd.isna(lat) or pd.isna(lon):
                    stats.rows_skipped_invalid_data += 1
                    pbar.update(1)
                    continue

                # Parse year from collection_date column
                date_str = row.get("collection_date", "")
                year = parse_date_to_year(str(date_str))

                if year is None:
                    stats.rows_skipped_invalid_data += 1
                    pbar.update(1)
                    continue

                # Track cache size before this request
                cache_before = get_ge_cache_stats()["total_cached"]

                # Get Google Earth Engine embedding
                try:
                    # Try original year first
                    try:
                        embedding = get_embedding(lat, lon, year, project)
                        df.at[idx, "google_earth_embeddings"] = str(embedding)  # type: ignore[index]

                        # Check if this was a cache hit
                        cache_after = get_ge_cache_stats()["total_cached"]
                        if cache_after == cache_before:
                            stats.cache_hits += 1
                        else:
                            stats.cache_misses += 1
                            stats.api_success += 1

                    except ValueError as e:
                        error_str = str(e)
                        # Check if this was a cached failure (instant)
                        is_cached_failure = "cached failure" in error_str

                        if "No embedding found" in error_str:
                            if is_cached_failure:
                                stats.cache_hits += (
                                    1  # Cached failure = instant, count as cache hit
                                )

                            # Try fallback year
                            try:
                                embedding = get_embedding(
                                    lat, lon, fallback_year, project
                                )
                                df.at[idx, "google_earth_embeddings"] = str(embedding)  # type: ignore[index]

                                # Check if this was a cache hit
                                cache_after = get_ge_cache_stats()["total_cached"]
                                if cache_after == cache_before + (
                                    1 if not is_cached_failure else 0
                                ):
                                    stats.cache_hits += 1
                                else:
                                    stats.cache_misses += 1
                                    stats.api_success += 1
                            except ValueError as fallback_e:
                                fallback_str = str(fallback_e)
                                is_fallback_cached = "cached failure" in fallback_str

                                if "No embedding found" in fallback_str:
                                    if is_fallback_cached and not is_cached_failure:
                                        stats.cache_hits += 1  # Fallback was cached
                                    stats.api_failures_no_coverage += 1
                                else:
                                    stats.api_failures_other += 1
                        else:
                            stats.api_failures_other += 1

                except Exception as e:
                    error_msg = str(e).lower()
                    if (
                        "429" in error_msg
                        or "too many requests" in error_msg
                        or "quota" in error_msg
                    ):
                        stats.api_failures_rate_limit += 1
                    else:
                        stats.api_failures_other += 1

                # Update progress bar
                pbar.set_postfix(
                    {
                        "Success": stats.total_processed(),
                        "Cache": stats.cache_hits,
                        "Failed": stats.total_failed(),
                    }
                )
                pbar.update(1)

    # Filter out rows with missing embeddings
    rows_before_filter = len(df)
    df_filtered = df.dropna(subset=["google_earth_embeddings"])
    rows_after_filter = len(df_filtered)
    rows_removed = rows_before_filter - rows_after_filter

    # Save the filtered file (same format as input)
    print(f"Saving results to {output_file}")
    df_filtered.to_csv(output_file, sep=delimiter, index=False)

    # Print detailed statistics
    stats.print_summary("Google Earth Embeddings Processing")

    # Cache stats
    cache_stats = get_ge_cache_stats()
    print("\nCache information:")
    print(f"  - Total entries in cache: {cache_stats['total_cached']}")
    print(f"  - Cache directory: {cache_stats['cache_dir']}")

    print("\nOutput filtering:")
    print(f"  - Rows before filtering: {rows_before_filter}")
    print(f"  - Rows after filtering: {rows_after_filter}")
    print(f"  - Rows removed (incomplete embeddings): {rows_removed}")

    return stats.total_processed()


def add_envo_embeddings_to_csv(
    csv_file: Path,
    output_file: Path,
    max_rows: Optional[int] = None,
    skip_existing: bool = True,
    random_sample: bool = True,
) -> int:
    """Add ENVO embeddings for all three environmental columns to CSV file.

    This function processes the three ENVO columns in Mark's normalized CSV:
    - env_broad_scale
    - env_medium
    - env_local_scale

    And creates three embedding columns:
    - envo_broad_scale_embedding
    - envo_medium_embedding
    - envo_local_scale_embedding

    Rows with incomplete ENVO embeddings are filtered out from the final output.
    Uses caching to avoid redundant API calls for duplicate ENVO terms.

    Args:
        csv_file: Path to input CSV file
        output_file: Path to output CSV file
        max_rows: Maximum number of rows to process (all rows if None)
        skip_existing: Skip rows that already have all three embeddings
        random_sample: If True, randomly sample max_rows (avoids geographic/temporal bias).
                      If False, take first max_rows sequentially (useful for resuming).
                      Default: True (recommended for representative sampling)

    Returns:
        Number of rows successfully processed with all embeddings

    Notes:
        Random sampling (random_sample=True):
        - ✅ Use for: Initial exploration, getting representative sample
        - ✅ Avoids bias from data ordering (geography, time, bioproject)
        - ✅ Better for statistical validity
        - Uses fixed seed (42) for reproducibility

        Sequential sampling (random_sample=False):
        - ✅ Use for: Resuming failed jobs, processing specific rows
        - ✅ Deterministic ordering (same rows every time)
        - ⚠️  May introduce bias if data is ordered (e.g., all samples from one region first)
    """
    # Auto-detect file format (CSV vs TSV)
    delimiter = "\t" if csv_file.suffix.lower() == ".tsv" else ","
    file_type = "TSV" if delimiter == "\t" else "CSV"

    # Read the file
    print(f"Loading {file_type} file: {csv_file}")
    df = pd.read_csv(csv_file, sep=delimiter, low_memory=False)

    if max_rows:
        if random_sample:
            # Random sample - better for representative sampling
            df = df.sample(n=min(max_rows, len(df)), random_state=42)
            print(
                f"Processing random sample of {len(df)} rows (seed=42 for reproducibility)"
            )
        else:
            # Sequential - first N rows
            df = df.head(max_rows)
            print(f"Processing first {len(df)} rows sequentially")

    # Add embedding columns if they don't exist
    for col_name in [
        "envo_broad_scale_embedding",
        "envo_medium_embedding",
        "envo_local_scale_embedding",
    ]:
        if col_name not in df.columns:
            df[col_name] = None
            print(f"Added '{col_name}' column")

    # Track detailed statistics
    stats = ProcessingStats()
    total_rows = len(df)

    print(f"Processing {total_rows} rows for ENVO embeddings...")
    print("Step 1: Collecting unique ENVO terms from all three columns...")

    # Collect all unique ENVO terms (optimized like Google Earth embeddings)
    unique_terms = set()
    term_to_column: dict = {}  # Track which column each term came from

    for col in ["env_broad_scale", "env_medium", "env_local_scale"]:
        for text in df[col].dropna():
            if isinstance(text, str) and text.strip():
                envo_term = extract_first_envo_term(text)
                if envo_term:
                    unique_terms.add(envo_term)
                    if envo_term not in term_to_column:
                        term_to_column[envo_term] = []
                    term_to_column[envo_term].append(col)

    print(f"  - Found {len(unique_terms)} unique ENVO terms")

    # Check cache and fetch uncached terms
    print("Step 2: Checking cache and fetching uncached terms...")
    from .envo_embeddings import _cache as envo_cache

    cached_terms = set()
    uncached_terms = set()
    term_embeddings = {}

    for term in unique_terms:
        if term in envo_cache:
            cached_terms.add(term)
            term_embeddings[term] = envo_cache[term]
            stats.cache_hits += 1
        else:
            uncached_terms.add(term)
            stats.cache_misses += 1

    print(f"  - {len(cached_terms)} terms in cache")
    print(f"  - {len(uncached_terms)} terms need fetching")

    # Fetch uncached terms with progress bar
    if uncached_terms:
        print("Step 3: Fetching uncached ENVO embeddings...")
        with tqdm(total=len(uncached_terms), desc="Fetching ENVO", unit="term") as pbar:
            for term in uncached_terms:
                embedding = get_envo_embedding(term)
                if embedding:
                    term_embeddings[term] = embedding
                    stats.api_success += 1
                else:
                    stats.api_failures_other += 1
                pbar.update(1)
    else:
        print("Step 3: All terms cached - skipping fetch")

    # Map embeddings back to dataframe (vectorized with pandas .map())
    print("Step 4: Mapping embeddings back to dataframe...")

    def get_embedding_str(text_value):
        """Extract ENVO term and return embedding as string, or None."""
        if (
            pd.isna(text_value)
            or not isinstance(text_value, str)
            or not text_value.strip()
        ):
            return None
        envo_term = extract_first_envo_term(text_value)
        if envo_term and envo_term in term_embeddings:
            return str(term_embeddings[envo_term])
        return None

    # Vectorized mapping for all three columns
    print("  - Mapping env_broad_scale...")
    df["envo_broad_scale_embedding"] = df["env_broad_scale"].apply(get_embedding_str)

    print("  - Mapping env_medium...")
    df["envo_medium_embedding"] = df["env_medium"].apply(get_embedding_str)

    print("  - Mapping env_local_scale...")
    df["envo_local_scale_embedding"] = df["env_local_scale"].apply(get_embedding_str)

    # Filter out rows with incomplete embeddings (must have all 4: GE + 3 ENVO)
    rows_before_filter = len(df)

    # Check if google_earth_embeddings column exists
    required_cols = [
        "envo_broad_scale_embedding",
        "envo_medium_embedding",
        "envo_local_scale_embedding",
    ]
    if "google_earth_embeddings" in df.columns:
        required_cols.append("google_earth_embeddings")
        print("\nFiltering: Requiring ALL embeddings (Google Earth + 3 ENVO)")
    else:
        print("\nWarning: No google_earth_embeddings column - only filtering on ENVO")

    df_filtered = df.dropna(subset=required_cols)
    rows_after_filter = len(df_filtered)
    rows_removed = rows_before_filter - rows_after_filter

    # Save as CSV (notebook format) regardless of input format
    print(f"Saving results to {output_file}")
    df_filtered.to_csv(output_file, index=False)
    print("Output format: CSV (notebook-compatible)")

    # Print detailed statistics
    stats.print_summary("ENVO Embeddings Processing")

    # Print final cache statistics
    cache_stats = get_cache_stats()
    print("\nCache information:")
    print(f"  - Total unique ENVO terms cached: {cache_stats['total_cached']}")
    print(f"  - Successful embeddings in cache: {cache_stats['successful']}")
    print(f"  - Failed lookups in cache: {cache_stats['failed']}")
    print(f"  - Cache directory: {cache_stats['cache_dir']}")

    print("\nOutput filtering:")
    print(f"  - Input rows: {rows_before_filter}")
    print(f"  - Complete rows: {rows_after_filter}")
    print(f"  - Removed (incomplete): {rows_removed}")

    return stats.total_processed()


def prepare_coordinates_for_ee_upload(
    input_file: Path,
    output_file: Path,
    max_rows: Optional[int] = None,
) -> tuple[Path, Path]:
    """Prepare coordinates CSV for Earth Engine bulk upload (optimized for unique coordinates).

    Creates two files:
    1. Coordinates CSV: Unique (lat, lon, year) for Earth Engine processing
    2. Mapping CSV: Links sample IDs to coordinate IDs for result expansion

    This optimized approach processes each unique coordinate only once on Earth Engine,
    then expands results back to all samples using the mapping file.

    Args:
        input_file: Path to input CSV/TSV with columns: accession, latitude, longitude, collection_date
        output_file: Path for output coordinates CSV (mapping CSV will be output_file.stem + '_mapping.csv')
        max_rows: Maximum rows to process from input (for testing)

    Returns:
        Tuple of (coordinates_file_path, mapping_file_path)

    Example:
        >>> coords, mapping = prepare_coordinates_for_ee_upload(
        ...     Path('data/nmdc.tsv'),
        ...     Path('data/nmdc_coords_for_ee.csv')
        ... )
        >>> # Upload coords to Earth Engine
        >>> # After processing, use merge_ee_results() with the mapping file
    """
    # Auto-detect file format
    delimiter = "\t" if input_file.suffix.lower() == ".tsv" else ","

    # Read input
    df = pd.read_csv(input_file, sep=delimiter, low_memory=False)

    if max_rows:
        df = df.head(max_rows)

    # Parse year from collection_date
    df["year"] = df["collection_date"].apply(parse_date_to_year)

    # Filter to valid rows
    df_valid = df[
        df["latitude"].notna() & df["longitude"].notna() & df["year"].notna()
    ].copy()

    print(f"Input: {len(df)} rows")
    print(f"Valid coordinates: {len(df_valid)} rows")

    # Create unique coordinate ID for each (lat, lon, year) combination
    # Round to 4 decimals to match cache normalization
    df_valid["lat_rounded"] = df_valid["latitude"].round(4)
    df_valid["lon_rounded"] = df_valid["longitude"].round(4)

    # Create coordinate tuple for grouping
    df_valid["coord_tuple"] = list(
        zip(df_valid["lat_rounded"], df_valid["lon_rounded"], df_valid["year"])
    )

    # Get unique coordinates
    unique_coords = df_valid[
        ["coord_tuple", "lat_rounded", "lon_rounded", "year"]
    ].drop_duplicates(subset="coord_tuple")

    # Assign coordinate IDs
    unique_coords = unique_coords.reset_index(drop=True)
    unique_coords["coord_id"] = "coord_" + unique_coords.index.astype(str).str.zfill(6)

    # Create coordinates CSV for Earth Engine upload
    coords_df = unique_coords[["coord_id", "lat_rounded", "lon_rounded", "year"]].copy()
    coords_df.columns = ["coord_id", "latitude", "longitude", "year"]  # type: ignore[assignment]

    # Create mapping from sample_id to coord_id
    # Merge coord_id back to all samples
    df_valid = df_valid.merge(
        unique_coords[["coord_tuple", "coord_id"]], on="coord_tuple", how="left"
    )

    mapping_df = df_valid[
        ["accession", "coord_id", "latitude", "longitude", "year"]
    ].copy()
    mapping_df.columns = ["sample_id", "coord_id", "latitude", "longitude", "year"]  # type: ignore[assignment]

    # Save both files
    coords_df.to_csv(output_file, index=False)
    mapping_file = output_file.parent / f"{output_file.stem}_mapping.csv"
    mapping_df.to_csv(mapping_file, index=False)

    print("\n✓ Created optimized coordinates for Earth Engine:")
    print(
        f"  - Unique coordinates: {len(coords_df)} (reduced from {len(df_valid)} samples)"
    )
    print(f"  - Reduction: {100 * (1 - len(coords_df) / len(df_valid)):.1f}%")
    print(f"  - Year range: {coords_df['year'].min()} - {coords_df['year'].max()}")
    print("\nOutput files:")
    print(f"  1. Coordinates CSV (upload to EE): {output_file}")
    print(f"  2. Mapping CSV (for result expansion): {mapping_file}")

    return output_file, mapping_file


def merge_ee_results(
    mapping_file: Path,
    ee_results_file: Path,
    original_file: Path,
    output_file: Path,
) -> int:
    """Merge Earth Engine results back to original samples using mapping file.

    Takes the unique coordinate results from Earth Engine and expands them
    back to all original samples using the coord_id mapping.

    Args:
        mapping_file: Path to mapping CSV (from prepare_coordinates_for_ee_upload)
        ee_results_file: Path to Earth Engine results CSV (has coord_id and embedding columns)
        original_file: Path to original input file
        output_file: Path for merged output file

    Returns:
        Number of samples successfully merged with embeddings

    Example:
        >>> # After Earth Engine processing completes
        >>> merge_ee_results(
        ...     Path('data/nmdc_coords_mapping.csv'),
        ...     Path('data/ee_results.csv'),
        ...     Path('data/nmdc.tsv'),
        ...     Path('data/nmdc_with_embeddings.tsv')
        ... )
    """
    # Auto-detect file formats
    original_delimiter = "\t" if original_file.suffix.lower() == ".tsv" else ","
    output_delimiter = "\t" if output_file.suffix.lower() == ".tsv" else ","

    # Load files
    print(f"Loading mapping file: {mapping_file}")
    mapping_df = pd.read_csv(mapping_file)

    print(f"Loading Earth Engine results: {ee_results_file}")
    ee_results = pd.read_csv(ee_results_file)

    print(f"Loading original file: {original_file}")
    original_df = pd.read_csv(original_file, sep=original_delimiter, low_memory=False)

    # Merge EE results to mapping (coord_id → embeddings)
    mapping_with_embeddings = mapping_df.merge(
        ee_results[["coord_id", "google_earth_embeddings"]], on="coord_id", how="left"
    )

    # Merge to original file (sample_id/accession → embeddings)
    result_df = original_df.merge(
        mapping_with_embeddings[["sample_id", "google_earth_embeddings"]],
        left_on="accession",
        right_on="sample_id",
        how="left",
    )

    # Drop the redundant sample_id column from merge
    if "sample_id" in result_df.columns:
        result_df = result_df.drop(columns=["sample_id"])

    # Filter out rows without Google Earth embeddings
    rows_before = len(result_df)
    result_df = result_df.dropna(subset=["google_earth_embeddings"])
    rows_after = len(result_df)
    rows_removed = rows_before - rows_after

    # Save filtered results
    result_df.to_csv(output_file, sep=output_delimiter, index=False)

    print("\n✓ Merge complete:")
    print(f"  - Input samples: {rows_before}")
    print(f"  - With embeddings: {rows_after}")
    print(f"  - Removed (no coverage): {rows_removed}")
    print(f"  - Success rate: {100 * rows_after / rows_before:.1f}%")
    print(f"\nOutput: {output_file}")

    return rows_after


if __name__ == "__main__":
    # Test the coordinate parsing
    test_coords = [
        "50.936 N 6.952 E",
        "35.32 S 148.25 E",
        "28.1000 N 81.6000 W",
        "41.34 N 2.10 E",
    ]

    for coord in test_coords:
        result = parse_coordinate_string(coord)
        print(f"{coord} -> {result}")

    # Test date parsing
    test_dates = ["2008-08-20", "2016", "1998-12", "1997-07-11"]
    for date in test_dates:
        date_result = parse_date_to_year(date)
        print(f"{date} -> {date_result}")
