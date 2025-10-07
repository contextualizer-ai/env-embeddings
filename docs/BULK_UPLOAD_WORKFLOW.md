# Earth Engine Bulk Upload Workflow

This workflow processes satellite embeddings using Earth Engine's server-side processing, avoiding client-side loops and dramatically improving performance.

## Performance Comparison

**Current approach (NMDC - 1828 unique coords):**
- One-by-one: ~12-13 minutes
- "Batched" (still one-by-one due to tiling): ~10-12 minutes

**Bulk upload approach (NMDC - 1828 unique coords):**
- Estimated: **3-5 minutes total**
  - Upload: 10-20 seconds
  - Server-side processing: 2-4 minutes (parallelized by Google)
  - Download: 5-10 seconds

**For larger datasets (NCBI - 300K unique coords):**
- Current: ~40 hours
- Bulk upload: **20-30 minutes**

## Step-by-Step Instructions

### 1. Prepare Coordinates

```bash
# Using justfile
just prepare-ee-coords

# Or directly
uv run env-embeddings prepare-ee-coords data/nmdc_flattened_biosample_for_env_embeddings_202510061052.tsv
```

**Output:**
- `data/nmdc_..._coords_for_ee.csv` - Unique coordinates (1828 rows)
- `data/nmdc_..._coords_for_ee_mapping.csv` - Mapping to expand results back

### 2. Process on Earth Engine

**For small datasets (< 2K coords):**

```bash
uv run env-embeddings bulk-process-ee \
  data/nmdc_flattened_biosample_for_env_embeddings_202510061052_coords_for_ee.csv \
  --description nmdc_embeddings
```

**For large datasets (> 10K coords), first upload as Earth Engine asset, then:**

```bash
uv run env-embeddings bulk-process-ee \
  data/ncbi_coords_for_ee.csv \
  --asset projects/env-embeddings-2025/assets/ncbi_coords \
  --description ncbi_embeddings
```

**Options:**
- `--export-to drive` - Export to Google Drive (default, easiest)
- `--export-to gcs --bucket YOUR_BUCKET` - Export to Google Cloud Storage (faster for large files)
- `--no-monitor` - Start task and exit (check status manually)
- `--asset ASSET_ID` - Use pre-uploaded Earth Engine Table asset (for large datasets)

**What happens:**
1. Uploads coordinates to Earth Engine as FeatureCollection
2. Configures server-side sampling (`.map()` over all points)
3. Starts export task
4. Monitors task until completion

### 3. Download Results

**From Google Drive:**
- Go to: https://drive.google.com
- Navigate to: `earth_engine_exports/` folder
- Download: `nmdc_embeddings.csv`

**From Google Cloud Storage:**
```bash
gsutil cp gs://YOUR_BUCKET/nmdc_embeddings.csv data/ee_results.csv
```

### 4. Convert Embeddings Format

The Earth Engine export has separate columns for each embedding dimension (A00, A01, ..., A63). We need to convert this to our format (single column with list).

```bash
uv run env-embeddings convert-ee-results data/ee_results.csv
```

**Output:** `data/ee_results_converted.csv`

### 5. Merge Results Back

```bash
# Using justfile
just merge-ee-results

# Or directly
uv run env-embeddings merge-ee-results-cmd \
  data/nmdc_flattened_biosample_for_env_embeddings_202510061052_coords_for_ee_mapping.csv \
  data/ee_results_converted.csv \
  data/nmdc_flattened_biosample_for_env_embeddings_202510061052.tsv
```

**Output:**
- `data/nmdc_flattened_biosample_for_env_embeddings_202510061052_with_embeddings.tsv`

## Troubleshooting

### Earth Engine Authentication

If you get authentication errors:

```bash
earthengine authenticate
```

Follow the prompts to authorize.

### Task Fails with "Memory limit exceeded"

For very large datasets (> 10K coordinates), use the Earth Engine asset upload workflow instead of processing in chunks. Upload your coordinates CSV as a Table asset via the Earth Engine Code Editor, then reference it with the `--asset` flag.

### Export Takes Too Long

Check task status at: https://code.earthengine.google.com/tasks

You can cancel monitoring (Ctrl+C) and the task will continue running on Google's servers.

## Files Created

```
data/
├── nmdc_..._coords_for_ee.csv           # 1828 unique coords → Upload to EE
├── nmdc_..._coords_for_ee_mapping.csv   # 8434 rows → For result expansion
├── ee_results.csv                        # Downloaded from Drive/GCS
├── ee_results_converted.csv             # Converted to our format
└── nmdc_..._with_embeddings.tsv         # Final output (8434 rows with embeddings)
```

## Next Steps

After getting embeddings, you can:

1. **Add ENVO embeddings:**
   ```bash
   uv run env-embeddings add-envo-embeddings-csv data/nmdc_..._with_embeddings.tsv
   ```

2. **Analyze bioproject quality:**
   ```bash
   just rank-bioprojects
   ```
