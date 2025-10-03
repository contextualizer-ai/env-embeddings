# Column Mapping: Old TSV to New CSV

## Overview

Mapping from `date_and_latlon_samples_extended_ONLY_ENVO_SAMPLE_100_with_both_embeddings.tsv` (old) to `satisfying_biosamples_normalized.csv` (new).

## Key Differences

### File Format
- **Old**: TSV (tab-separated)
- **New**: CSV (comma-separated)

### Dataset Size
- **Old**: 99 rows
- **New**: 436,337 rows (much larger dataset)

## Column Mappings

### Date Fields

| Old Column | New Column | Notes |
|------------|------------|-------|
| `date` | `collection_date` | Normalized date format |
| `collection_date` | `collection_date_raw` | Raw date from source |

**Format**: Both use ISO format (YYYY-MM-DD or YYYY-MM)

### Location Fields

| Old Column | New Column | Notes |
|------------|------------|-------|
| `lat_lon` | `lat_lon_raw` | Combined lat/lon string (e.g., "35.118 N 138.937 E") |
| `ncbi_lat_lon` | `lat_lon_raw` | NCBI version of lat/lon string |
| N/A | `latitude` | **NEW: Separate numeric latitude** |
| N/A | `longitude` | **NEW: Separate numeric longitude** |

**Key Difference**:
- **Old file** stores latitude and longitude in a single column as a formatted string (e.g., "42.36 N 71.06 W")
- **New file** has both:
  - `lat_lon_raw`: Original formatted string
  - `latitude`: Numeric latitude (e.g., 35.118)
  - `longitude`: Numeric longitude (e.g., 138.937)

### Accession/ID Fields

| Old Column | New Column | Notes |
|------------|------------|-------|
| `ncbi_biosample_accession_id` | `accession` | Biosample accession ID |
| `genome_id` | N/A | **MISSING in new file** |

### Environmental Fields

| Old Column | New Column | Notes |
|------------|------------|-------|
| `env_broad_med_local` | `env_broad_scale`, `env_local_scale`, `env_medium` | **Split into 3 separate ENVO columns** |

**Key Difference**:
- **Old file**: Combined environmental ontology terms in one column
- **New file**: Separate columns for broad scale, local scale, and medium ENVO terms

### Missing Columns in New File

These columns exist in the old file but are **NOT present** in the new file:
- `genome_id`
- `ncbi_bioproject`
- `domain`, `phylum`, `class`, `order`, `family`, `genus`, `species` (taxonomic hierarchy)
- `geographic_location_harmonized`
- `host_harmonized`
- `isolation_source_harmonized`
- `project_name`
- `misc_attributes`
- `google_earth_embeddings`
- `envo_embeddings`

## Notes for Integration

1. **Embeddings will need to be generated** for the new file (google_earth_embeddings and envo_embeddings)
2. **Latitude/longitude parsing** is already done in new file (no need to parse string format)
3. **Genome ID missing** - may need to join with another data source if genome info is required
4. **Environmental terms are more structured** in new file (separate ENVO columns)
5. **Much larger dataset** (436k vs 99 rows) - may impact processing time
