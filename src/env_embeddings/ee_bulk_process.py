#!/usr/bin/env python3
"""Earth Engine bulk processing script for satellite embeddings.

This script uploads coordinates to Earth Engine, processes them server-side,
and exports results to Google Drive or Cloud Storage.

Usage:
    python scripts/ee_bulk_process.py \\
        --coords data/nmdc_coords_for_ee.csv \\
        --project env-embeddings-2025 \\
        --export-to drive \\
        --description nmdc_embeddings
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Optional

import ee


def initialize_earth_engine(project: str):
    """Initialize Earth Engine with project ID."""
    try:
        ee.Initialize(project=project)
        print(f"✓ Earth Engine initialized with project: {project}")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        print("Run: earthengine authenticate")
        raise


def upload_coordinates_to_ee(coords_file: Path, asset_id: str) -> ee.FeatureCollection:
    """Upload coordinates CSV to Earth Engine as FeatureCollection.

    Args:
        coords_file: Path to coordinates CSV (coord_id, latitude, longitude, year)
        asset_id: Earth Engine asset ID (e.g., 'users/YOUR_USERNAME/nmdc_coords')

    Returns:
        FeatureCollection with uploaded coordinates
    """
    print(f"\nReading coordinates from: {coords_file}")

    # Read CSV and create features
    features = []
    with open(coords_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coord_id = row["coord_id"]
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            year = int(float(row["year"]))  # Handle "2021.0" format

            # Create point geometry
            point = ee.Geometry.Point([lon, lat])

            # Create feature with properties
            feature = ee.Feature(
                point,
                {"coord_id": coord_id, "year": year, "latitude": lat, "longitude": lon},
            )
            features.append(feature)

    print(f"  - Loaded {len(features)} unique coordinates")

    # Create FeatureCollection
    fc = ee.FeatureCollection(features)

    print("  - Created FeatureCollection")

    # Note: For large datasets, you may need to upload via Asset Manager
    # For now, we'll use the in-memory FeatureCollection

    return fc


def sample_embeddings(coords_fc: ee.FeatureCollection) -> ee.FeatureCollection:
    """Sample satellite embeddings for each coordinate (server-side processing).

    Args:
        coords_fc: FeatureCollection with coordinates and year properties

    Returns:
        FeatureCollection with embeddings added as properties
    """
    print("\nSetting up server-side sampling...")

    def sample_point(feature):
        """Sample embedding for a single point (runs on EE servers)."""
        point = feature.geometry()
        year = ee.Number(feature.get("year")).toInt()

        # Get the embedding image for this year and location
        start = ee.Date.fromYMD(year, 1, 1)
        end = start.advance(1, "year")

        # Filter to images that cover this point
        collection = (
            ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            .filterDate(start, end)
            .filterBounds(point)
        )

        # Check if collection has any images
        image = collection.first()

        # Use ee.Algorithms.If to handle null case (runs server-side)
        def process_with_image():
            sampled = image.sample(region=point, scale=10).first()
            # Return feature with embedding or mark as failed
            return ee.Algorithms.If(
                sampled,
                feature.set(sampled.toDictionary()),
                feature.set("no_coverage", True),
            )

        def process_no_image():
            return feature.set("no_coverage", True)

        return ee.Algorithms.If(
            collection.size().gt(0), process_with_image(), process_no_image()
        )

    # Map over all points (THIS RUNS ON GOOGLE'S SERVERS)
    results = coords_fc.map(sample_point)

    print("  - Server-side sampling function configured")

    return results


def export_results(
    results_fc: ee.FeatureCollection,
    description: str,
    export_to: str = "drive",
    folder: str = "earth_engine_exports",
    bucket: Optional[str] = None,
) -> ee.batch.Task:
    """Export results to Drive or Cloud Storage.

    Args:
        results_fc: FeatureCollection with embeddings
        description: Export task description
        export_to: 'drive' or 'gcs' (Google Cloud Storage)
        folder: Folder name for Drive export
        bucket: GCS bucket name (required if export_to='gcs')

    Returns:
        Export task
    """
    print(f"\nConfiguring export to {export_to.upper()}...")

    if export_to == "drive":
        task = ee.batch.Export.table.toDrive(
            collection=results_fc,
            description=description,
            folder=folder,
            fileFormat="CSV",
            selectors=["coord_id", "year", "latitude", "longitude"]
            + [f"A{str(i).zfill(2)}" for i in range(64)],
        )
        print(f"  - Export to Google Drive folder: {folder}")

    elif export_to == "gcs":
        if not bucket:
            raise ValueError("bucket parameter required for GCS export")

        task = ee.batch.Export.table.toCloudStorage(
            collection=results_fc,
            description=description,
            bucket=bucket,
            fileNamePrefix=description,
            fileFormat="CSV",
            selectors=["coord_id", "year", "latitude", "longitude"]
            + [f"A{str(i).zfill(2)}" for i in range(64)],
        )
        print(f"  - Export to GCS bucket: gs://{bucket}/{description}")

    else:
        raise ValueError(f"Invalid export_to: {export_to} (must be 'drive' or 'gcs')")

    # Start the task
    task.start()
    print(f"  - Task started: {description}")
    print(f"  - Task ID: {task.id}")

    return task


def monitor_task(task: ee.batch.Task, check_interval: int = 30):
    """Monitor export task until completion.

    Args:
        task: Export task to monitor
        check_interval: Seconds between status checks
    """
    print(f"\nMonitoring task (checking every {check_interval}s)...")
    print("  Press Ctrl+C to stop monitoring (task will continue running)")

    try:
        while True:
            status = task.status()
            state = status["state"]

            if state == "COMPLETED":
                print("\n✅ Task completed successfully!")
                if "destination_uris" in status:
                    print(f"  Download from: {status['destination_uris']}")
                return True

            elif state == "FAILED":
                print("\n❌ Task failed!")
                if "error_message" in status:
                    print(f"  Error: {status['error_message']}")
                return False

            elif state in ["READY", "RUNNING"]:
                print(f"  Status: {state}...", end="\r")
                time.sleep(check_interval)

            else:
                print(f"  Unknown state: {state}")
                time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Task is still running.")
        print("Check status at: https://code.earthengine.google.com/tasks")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Bulk process satellite embeddings using Earth Engine"
    )
    parser.add_argument(
        "--coords", type=Path, required=True, help="Path to coordinates CSV"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="env-embeddings-2025",
        help="Google Cloud project ID",
    )
    parser.add_argument(
        "--export-to",
        choices=["drive", "gcs"],
        default="drive",
        help="Export destination (drive or gcs)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="satellite_embeddings",
        help="Export task description",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="earth_engine_exports",
        help="Drive folder name (for --export-to drive)",
    )
    parser.add_argument(
        "--bucket", type=str, help="GCS bucket name (for --export-to gcs)"
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Start task but don't monitor (check manually)",
    )
    parser.add_argument(
        "--asset",
        type=str,
        help="Use existing Earth Engine asset instead of uploading CSV (e.g., projects/PROJECT/assets/TABLE)",
    )

    args = parser.parse_args()

    # Initialize Earth Engine
    initialize_earth_engine(args.project)

    # Load coordinates - from asset if provided, otherwise upload CSV
    if args.asset:
        print(f"\nLoading coordinates from Earth Engine asset: {args.asset}")
        coords_fc = ee.FeatureCollection(args.asset)
        count = coords_fc.size().getInfo()
        print(f"  - Loaded {count} coordinates from asset")
    else:
        # Upload coordinates (original behavior)
        coords_fc = upload_coordinates_to_ee(
            args.coords, f"users/tmp/{args.description}"
        )

    # Process embeddings (server-side)
    results_fc = sample_embeddings(coords_fc)

    # Export
    task = export_results(
        results_fc,
        description=args.description,
        export_to=args.export_to,
        folder=args.folder,
        bucket=args.bucket,
    )

    # Monitor
    if not args.no_monitor:
        monitor_task(task)
    else:
        print("\nTask started. Check status at:")
        print("  https://code.earthengine.google.com/tasks")
        print(f"\nTask ID: {task.id}")


if __name__ == "__main__":
    main()
