# Data File Analysis - 2025-10-03

## Latest Data File: `satisfying_biosamples_normalized_with_google_embeddings_with_envo_embeddings.csv`

**Last Modified:** Oct 3 05:44 (2025)

**Size:** 3.5 GB

### Distinguishing Characteristics

- **34,129 rows** (from 436,337 in the input - only ~7.8% passed)
- Contains **both Google Earth embeddings AND ENVO embeddings**
- Columns:
  - accession
  - collection_date
  - env_broad_scale
  - env_local_scale
  - env_medium
  - collection_date_raw
  - lat_lon_raw
  - latitude
  - longitude
  - google_earth_embeddings
  - envo_broad_scale_embedding
  - envo_medium_embedding
  - envo_local_scale_embedding

### Generated From

- **Input:** `data/satisfying_biosamples_normalized.csv` (436,337 rows, normalized biosample data with environmental context)
- **Process:** Output of adding both Google Earth embeddings and ENVO embeddings to the normalized biosamples

### Usage

Currently referenced in `notebooks/similarity_analysis.ipynb:78` for similarity analysis between Google and ENVO embeddings.

### Notes

This is the most complete dataset with both embedding types for comparison analysis.

## File Cleanup Analysis

### Safe to Delete (not referenced or deprecated):

1. `data/satisfying_biosamples_normalized_with_google_embeddings.csv` - Intermediate file, superseded by the version with both embeddings
2. `data/satisfying_biosamples_normalized_with_google_embeddings_with_envo_embeddings.csv` - Only used in old notebook, likely superseded by `satisfying_biosamples_normalized.csv`

### Keep - Actively Used:

1. ✅ `tests/input/example-samples.tsv` - Git tracked, used for tests
2. ✅ `data/satisfying_biosamples_normalized.csv` - Git tracked, referenced in justfile and docs
3. ✅ `data/biosamples_with_embeddings.csv` - Git tracked, used in `bioproject_quality.py:282` and BIOPROJECT_QUALITY.md

### Keep - Sample Data (Git tracked):

4. ✅ `data/date_and_latlon_samples_extended_SAMPLE.tsv` - Git tracked sample
5. ✅ `data/date_and_latlon_samples_extended_SAMPLE_with_embeddings.tsv` - Git tracked sample
6. ✅ `data/date_and_latlon_samples_extended_ONLY_ENVO_SAMPLE_100.tsv` - Git tracked sample
7. ✅ `data/date_and_latlon_samples_extended_ONLY_ENVO_SAMPLE_100_with_embeddings.tsv` - Git tracked sample
8. ✅ `data/date_and_latlon_samples_extended_ONLY_ENVO_SAMPLE_100_with_both_embeddings.tsv` - Git tracked, mentioned in docs/research/column_mapping.md
