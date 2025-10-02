## Add your own just recipes here. This is imported by the main justfile.

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
