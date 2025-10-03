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
)
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


def parse_date_to_year(date_str: str) -> Optional[int]:
    """Parse various date formats to extract year.

    Args:
        date_str: Date string in various formats

    Returns:
        Year as integer, or None if invalid

    Examples:
        >>> parse_date_to_year("2008-08-20")
        2008
        >>> parse_date_to_year("2016")
        2016
        >>> parse_date_to_year("1998-12")
        1998
    """
    if not date_str or date_str.strip() == "":
        return None

    try:
        # Extract first 4-digit number (assuming it's the year)
        year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
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


def add_google_earth_embeddings_to_csv(
    csv_file: Path,
    output_file: Path,
    max_rows: Optional[int] = None,
    project: str = "env-embeddings-2025",
    fallback_year: int = 2020,
    skip_existing: bool = True,
    random_sample: bool = True,
) -> int:
    """Add Google Earth Engine embeddings to Mark's normalized CSV file.

    This function is specifically designed for the new CSV format with:
    - collection_date (normalized date column)
    - latitude, longitude (numeric columns)

    Rows with missing embeddings are filtered out from the final output.

    Args:
        csv_file: Path to input CSV file
        output_file: Path to output CSV file
        max_rows: Maximum number of rows to process (all rows if None)
        project: Google Cloud project ID
        fallback_year: Year to use when original year has no satellite data
        skip_existing: Skip rows that already have embeddings
        random_sample: If True, randomly sample max_rows (avoids geographic/temporal bias).
                      If False, take first max_rows sequentially (useful for resuming).
                      Default: True (recommended for representative sampling)

    Returns:
        Number of rows successfully processed with embeddings

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

    # Read the CSV file
    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file, low_memory=False)

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
                            embedding = get_embedding(lat, lon, fallback_year, project)
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

    # Save the filtered CSV
    print(f"Saving results to {output_file}")
    df_filtered.to_csv(output_file, index=False)

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
    # Read the CSV file
    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file, low_memory=False)

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

    # Note: Cache hit tracking is done per-request by comparing cache size before/after

    print(f"Processing {total_rows} rows for ENVO embeddings...")
    print("Note: Using cache to avoid redundant API calls for duplicate ENVO terms")

    # Create progress bar
    with tqdm(total=total_rows, desc="Adding ENVO embeddings", unit="row") as pbar:
        for idx, row in df.iterrows():
            # Skip if already has all embeddings and skip_existing is True
            if skip_existing and all(
                [
                    pd.notna(row.get("envo_broad_scale_embedding")),
                    pd.notna(row.get("envo_medium_embedding")),
                    pd.notna(row.get("envo_local_scale_embedding")),
                ]
            ):
                stats.rows_skipped_existing += 1
                pbar.update(1)
                continue

            # Process env_broad_scale
            broad_scale = row.get("env_broad_scale", "")
            if broad_scale and isinstance(broad_scale, str) and broad_scale.strip():
                envo_term = extract_first_envo_term(broad_scale)
                if envo_term:
                    cache_before = get_cache_stats()["total_cached"]
                    embedding = get_envo_embedding(envo_term)
                    cache_after = get_cache_stats()["total_cached"]

                    if embedding:
                        df.at[idx, "envo_broad_scale_embedding"] = str(embedding)  # type: ignore[index]

                        # Track cache hit/miss
                        if cache_after == cache_before:
                            stats.cache_hits += 1
                        else:
                            stats.cache_misses += 1
                            stats.api_success += 1
                    else:
                        stats.api_failures_other += 1
                else:
                    pass
            else:
                pass

            # Process env_medium
            medium = row.get("env_medium", "")
            if medium and isinstance(medium, str) and medium.strip():
                envo_term = extract_first_envo_term(medium)
                if envo_term:
                    cache_before = get_cache_stats()["total_cached"]
                    embedding = get_envo_embedding(envo_term)
                    cache_after = get_cache_stats()["total_cached"]

                    if embedding:
                        df.at[idx, "envo_medium_embedding"] = str(embedding)  # type: ignore[index]

                        # Track cache hit/miss
                        if cache_after == cache_before:
                            stats.cache_hits += 1
                        else:
                            stats.cache_misses += 1
                            stats.api_success += 1
                    else:
                        stats.api_failures_other += 1
                else:
                    pass
            else:
                pass

            # Process env_local_scale
            local_scale = row.get("env_local_scale", "")
            if local_scale and isinstance(local_scale, str) and local_scale.strip():
                envo_term = extract_first_envo_term(local_scale)
                if envo_term:
                    cache_before = get_cache_stats()["total_cached"]
                    embedding = get_envo_embedding(envo_term)
                    cache_after = get_cache_stats()["total_cached"]

                    if embedding:
                        df.at[idx, "envo_local_scale_embedding"] = str(embedding)  # type: ignore[index]

                        # Track cache hit/miss
                        if cache_after == cache_before:
                            stats.cache_hits += 1
                        else:
                            stats.cache_misses += 1
                            stats.api_success += 1
                    else:
                        stats.api_failures_other += 1
                else:
                    pass
            else:
                pass

            # Update progress bar
            pbar.set_postfix(
                {
                    "Success": stats.total_processed(),
                    "Cache": stats.cache_hits,
                    "Failed": stats.total_failed(),
                }
            )
            pbar.update(1)

    # Filter out rows with incomplete ENVO embeddings (must have all 3)
    rows_before_filter = len(df)
    df_filtered = df.dropna(
        subset=[
            "envo_broad_scale_embedding",
            "envo_medium_embedding",
            "envo_local_scale_embedding",
        ]
    )
    rows_after_filter = len(df_filtered)
    rows_removed = rows_before_filter - rows_after_filter

    # Save the filtered CSV
    print(f"Saving results to {output_file}")
    df_filtered.to_csv(output_file, index=False)

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
    print(f"  - Rows before filtering: {rows_before_filter}")
    print(f"  - Rows after filtering: {rows_after_filter}")
    print(f"  - Rows removed (incomplete embeddings): {rows_removed}")

    return stats.total_processed()


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
