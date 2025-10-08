"""Quick test of RF analysis module with small subset of data."""

from pathlib import Path
from src.env_embeddings.rf_analysis import (
    load_source_data,
    analyze_source,
    create_comparison_table,
    print_summary,
)

# Test with NMDC first (smaller file)
print("Testing RF analysis module...")
print("=" * 60)

nmdc_file = Path(
    "data/nmdc_flattened_biosample_for_env_embeddings_202510061052_complete.csv"
)

# Load data
df = load_source_data(nmdc_file, "NMDC", deduplicate=True)

if df is not None:
    print(f"\n✓ Successfully loaded {len(df)} samples")

    # Take small subset for quick test
    df_test = df.head(1000)
    print(f"\nTesting with subset of {len(df_test)} samples")

    # Analyze
    results = analyze_source(df_test, "NMDC-TEST")

    # Create comparison table
    all_results = {"NMDC-TEST": results}
    comparison_df = create_comparison_table(all_results)

    print("\n" + "=" * 60)
    print("Results:")
    print(comparison_df.to_string(index=False))

    print_summary(comparison_df)

    print("\n✓ Module test passed!")
else:
    print("✗ Failed to load data")
