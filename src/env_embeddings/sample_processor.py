"""Process sample data and retrieve Earth Engine embeddings."""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd

from .earth_engine import get_embedding, initialize_ee


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
        pattern = r'([0-9.-]+)\s*([NSEW])\s+([0-9.-]+)\s*([NSEW])'
        match = re.match(pattern, coord_str.strip())
        
        if not match:
            return None
            
        lat_val, lat_dir, lon_val, lon_dir = match.groups()
        
        lat = float(lat_val)
        lon = float(lon_val)
        
        # Apply direction multipliers
        if lat_dir.upper() == 'S':
            lat = -lat
        if lon_dir.upper() == 'W':
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
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
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
    df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
    
    if max_samples:
        df = df.head(max_samples)
    
    # Parse coordinates
    coord_data = df['lat_lon'].apply(parse_coordinate_string)
    df['parsed_lat'] = coord_data.apply(lambda x: x[0] if x else None)
    df['parsed_lon'] = coord_data.apply(lambda x: x[1] if x else None)
    
    # Parse dates to years
    df['parsed_year'] = df['date'].apply(parse_date_to_year)
    
    # Filter out samples with missing critical data
    valid_samples = df.dropna(subset=['parsed_lat', 'parsed_lon', 'parsed_year'])
    
    print(f"Loaded {len(df)} samples, {len(valid_samples)} have valid coordinates and dates")
    
    return valid_samples


def get_sample_embedding(sample: pd.Series, project: str = None, fallback_year: int = 2020) -> Optional[List[float]]:
    """Get Earth Engine embedding for a single sample.
    
    Args:
        sample: Pandas Series with parsed_lat, parsed_lon, parsed_year
        project: Google Cloud project ID
        fallback_year: Year to use if original year fails (default 2020)
        
    Returns:
        64-dimensional embedding vector or None if failed
    """
    try:
        lat = sample['parsed_lat']
        lon = sample['parsed_lon'] 
        year = int(sample['parsed_year'])
        
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
        print(f"Error getting embedding for sample {sample.get('genome_id', 'unknown')}: {e}")
        return None


def process_samples_batch(
    samples: pd.DataFrame, 
    project: str = None,
    output_path: Optional[Path] = None
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
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(samples)} samples, {successful_count} successful")
    
    # Add embeddings to dataframe
    samples_with_embeddings = samples.copy()
    samples_with_embeddings['earth_engine_embedding'] = embeddings
    
    # Filter to only successful embeddings
    successful_samples = samples_with_embeddings.dropna(subset=['earth_engine_embedding'])
    
    print(f"Successfully retrieved embeddings for {len(successful_samples)}/{len(samples)} samples")
    
    if output_path:
        successful_samples.to_csv(output_path, sep='\t', index=False)
        print(f"Results saved to {output_path}")
    
    return successful_samples


def add_embeddings_to_tsv(
    tsv_file: Path,
    output_file: Path,
    max_rows: Optional[int] = None,
    project: str = "env-embeddings-2025",
    fallback_year: int = 2020,
    skip_existing: bool = True
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
    df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
    
    if max_rows:
        df = df.head(max_rows)
        print(f"Processing first {max_rows} rows for testing")
    
    # Add google_earth_embeddings column if it doesn't exist
    if 'google_earth_embeddings' not in df.columns:
        df['google_earth_embeddings'] = None
        print("Added 'google_earth_embeddings' column")
    
    success_count = 0
    total_rows = len(df)
    
    print(f"Processing {total_rows} rows...")
    
    for idx, row in df.iterrows():
        # Skip if already has embeddings and skip_existing is True
        if skip_existing and pd.notna(row.get('google_earth_embeddings')):
            continue
            
        # Parse coordinates from lat_lon column
        lat_lon_str = row.get('lat_lon', '')
        coords = parse_coordinate_string(lat_lon_str)
        
        if coords is None:
            print(f"Row {idx+1}: Invalid coordinates '{lat_lon_str}'")
            continue
            
        lat, lon = coords
        
        # Parse year from date column
        date_str = row.get('date', '')
        year = parse_date_to_year(date_str)
        
        if year is None:
            print(f"Row {idx+1}: Invalid date '{date_str}'")
            continue
        
        # Get embedding
        try:
            # Try original year first
            try:
                embedding = get_embedding(lat, lon, year, project)
                df.at[idx, 'google_earth_embeddings'] = str(embedding)
                success_count += 1
                print(f"Row {idx+1}: Got embedding for {lat},{lon} in {year}")
                
            except ValueError as e:
                if "No embedding found" in str(e):
                    # Try fallback year
                    print(f"Row {idx+1}: No embedding for {lat},{lon} in {year}, trying {fallback_year}")
                    embedding = get_embedding(lat, lon, fallback_year, project)
                    df.at[idx, 'google_earth_embeddings'] = str(embedding)
                    success_count += 1
                    print(f"Row {idx+1}: Got embedding for {lat},{lon} in {fallback_year} (fallback)")
                else:
                    print(f"Row {idx+1}: Error with coordinates {lat},{lon}: {e}")
                    
        except Exception as e:
            print(f"Row {idx+1}: Failed to get embedding for {lat},{lon}: {e}")
        
        # Progress update every 50 rows
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{total_rows} rows, {success_count} successful embeddings")
    
    # Save the updated TSV
    print(f"Saving results to {output_file}")
    df.to_csv(output_file, sep='\t', index=False)
    
    print(f"Completed! Successfully added embeddings to {success_count}/{total_rows} rows")
    return success_count


if __name__ == "__main__":
    # Test the coordinate parsing
    test_coords = [
        "50.936 N 6.952 E",
        "35.32 S 148.25 E", 
        "28.1000 N 81.6000 W",
        "41.34 N 2.10 E"
    ]
    
    for coord in test_coords:
        result = parse_coordinate_string(coord)
        print(f"{coord} -> {result}")
        
    # Test date parsing
    test_dates = ["2008-08-20", "2016", "1998-12", "1997-07-11"]
    for date in test_dates:
        result = parse_date_to_year(date)
        print(f"{date} -> {result}")