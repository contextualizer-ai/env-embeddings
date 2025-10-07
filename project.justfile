## Add your own just recipes here. This is imported by the main justfile.

# Normalize NCBI raw biosample data
[group('embeddings')]
normalize-ncbi input='data/ncbi_flattened_biosamples_for_env_embeddings_202510061108.tsv':
  @echo "Normalizing NCBI biosample data..."
  uv run env-embeddings normalize-biosamples {{input}}
  @echo "✅ Done! Check data/ for normalized output"

# Add both Google Earth and ENVO embeddings (50 rows)
[group('embeddings')]
add-both-embeddings:
  @echo "Step 1: Adding Google Earth embeddings (50 rows)..."
  uv run env-embeddings add-google-embeddings-csv data/satisfying_biosamples_normalized.csv \
    --max-rows 50 \
    --output data/temp_with_google.csv
  @echo "Step 2: Adding ENVO embeddings (50 rows)..."
  uv run env-embeddings add-envo-embeddings-csv data/temp_with_google.csv \
    --max-rows 50 \
    --output data/with_both_embeddings.csv
  @echo "✅ Done! Output: data/with_both_embeddings.csv"

# Rank bioprojects by annotation quality
[group('quality-analysis')]
rank-bioprojects csv_file='data/with_both_embeddings.csv' envo_scale='envo_broad_scale':
  @echo "Ranking bioprojects from {{csv_file}}..."
  uv run env-embeddings rank-bioprojects {{csv_file}} \
    --envo-scale {{envo_scale}} \
    --mongo-host localhost \
    --mongo-port 27017
  @echo "✅ Done! Check data/ for output"

# Prepare coordinates for Earth Engine bulk upload
[group('embeddings')]
prepare-ee-coords input='data/nmdc_flattened_biosample_for_env_embeddings_202510061052.tsv':
  @echo "Preparing optimized coordinates for Earth Engine bulk upload..."
  uv run env-embeddings prepare-ee-coords {{input}}
  @echo "✅ Done! Upload the coords CSV to Earth Engine"

# Convert Earth Engine results to our format
[group('embeddings')]
convert-ee-results input='nmdc_embeddings.csv':
  @echo "Converting Earth Engine results format..."
  uv run env-embeddings convert-ee-results {{input}}
  @echo "✅ Done! Check data/ for converted output"

# Merge Earth Engine results back to original samples
[group('embeddings')]
merge-ee-results mapping='data/nmdc_flattened_biosample_for_env_embeddings_202510061052_coords_for_ee_mapping.csv' ee_results='data/nmdc_embeddings_converted.csv' original='data/nmdc_flattened_biosample_for_env_embeddings_202510061052.tsv':
  @echo "Merging Earth Engine results..."
  uv run env-embeddings merge-ee-results-cmd {{mapping}} {{ee_results}} {{original}}
  @echo "✅ Done! Check data/ for output"
