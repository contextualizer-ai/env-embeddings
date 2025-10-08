# MIxS Environmental Triad Ontology Term Analysis

## Overview

Script for analyzing ontology terms (primarily ENVO) used in MIxS environmental context triad fields.

The **MIxS environmental triad** consists of three fields that provide hierarchical environmental context:
- **env_broad_scale**: Broadest environmental context (typically biome, e.g., ENVO:00000446 "terrestrial biome")
- **env_local_scale**: Local environmental context (e.g., ENVO:00000114 "agricultural field")
- **env_medium**: Environmental material/medium (e.g., ENVO:00001998 "soil")

This script (`mixs_triad_analysis.py`) uses percentage-based thresholds and provides ML-relevant statistics for these fields across GOLD, NCBI, and NMDC data sources.

## Key Changes

### 1. Percentage-Based Thresholds (Primary)

**Before**: Absolute count thresholds (e.g., `min_frequency=5`)
- Problem: 5 occurrences in 11K records (0.045%) vs. 5 in 452K (0.001%) are very different
- Not meaningful for cross-dataset comparison

**After**: Percentage-based thresholds (e.g., `min_pct=0.1` for 0.1%)
- Scales appropriately to dataset size
- Meaningful across GOLD (11K), NCBI (452K), and NMDC (8K) datasets
- Still supports absolute override via `--min-absolute` if needed

**Default**: 0.1% (suitable for ML training - ensures terms appear at least 0.1% of time)

### 2. Enhanced Display

**Added Bottom N Terms Table**
- Shows rarest terms that passed the threshold
- Helps identify borderline cases for ML training

**Class Imbalance Analysis**
- Shows how many terms account for 50%, 80%, 95% of all occurrences
- Example: "6 terms (2.5% of unique terms) account for 80% of occurrences"
- Critical for understanding if dataset is dominated by few terms

**Multiple Threshold Statistics**
- Terms below 0.01%, 0.1%, 0.5%, 1.0%
- Helps decide appropriate cutoff for ML training

### 3. Return Values Updated

**Before**: Returns single DataFrame
```python
result = analyze_term_frequency(df, column, min_frequency)
```

**After**: Returns tuple of (DataFrame, statistics dict)
```python
result_df, stats = analyze_term_frequency(df, column, min_pct, min_absolute)
```

Statistics dict includes:
- `total_records`: Total rows in dataset
- `total_unique_terms`: Number of distinct ENVO terms
- `terms_after_filter`: Terms passing threshold
- `effective_min_count`: Actual minimum count used
- `terms_below_X%`: Counts of rare terms at various thresholds
- `terms_for_50pct`, `terms_for_80pct`, `terms_for_95pct`: Cumulative coverage

### 4. CLI Changes

#### frequency command
```bash
# Percentage-based threshold (default 0.1%)
uv run mixs-triad-analysis frequency \
  --file data/gold_flattened_biosamples_for_env_embeddings_202510061108.tsv \
  --source gold \
  --min-pct 0.1

# With absolute override
uv run mixs-triad-analysis frequency \
  --file data/ncbi_flattened_biosamples_for_env_embeddings_202510061108_normalized.tsv \
  --source ncbi \
  --min-pct 0.1 \
  --min-absolute 50
```

#### analyze-all command
```bash
# Analyze all three sources with percentage threshold
uv run mixs-triad-analysis analyze-all \
  --gold data/gold_flattened_biosamples_for_env_embeddings_202510061108.tsv \
  --ncbi data/ncbi_flattened_biosamples_for_env_embeddings_202510061108_normalized.tsv \
  --nmdc data/nmdc_flattened_biosample_for_env_embeddings_202510061052.tsv \
  --min-pct 0.1
```

### 5. Output Files

**Added**: JSON stats files alongside CSV frequency files
- `{source}_{column}_frequency.csv` - term frequencies (same as before)
- `{source}_{column}_stats.json` - comprehensive statistics (NEW)

## Usage Examples

### Analyze with 0.5% threshold (stricter - only frequent terms)
```bash
uv run mixs-triad-analysis frequency \
  --file data/gold_flattened_biosamples_for_env_embeddings_202510061108.tsv \
  --source gold \
  --min-pct 0.5
```

### Analyze with 0.01% threshold (permissive - include rare terms)
```bash
uv run mixs-triad-analysis frequency \
  --file data/ncbi_flattened_biosamples_for_env_embeddings_202510061108_normalized.tsv \
  --source ncbi \
  --min-pct 0.01
```

### Analyze all three sources
```bash
uv run mixs-triad-analysis analyze-all \
  --gold data/gold_flattened_biosamples_for_env_embeddings_202510061108.tsv \
  --ncbi data/ncbi_flattened_biosamples_for_env_embeddings_202510061108_normalized.tsv \
  --nmdc data/nmdc_flattened_biosample_for_env_embeddings_202510061052.tsv \
  --min-pct 0.1
```

### Parent analysis with exclusions

**Note**: Exclusions only filter out the specified parent terms from results. Subclasses of excluded terms are still shown with their counts. This is useful for hiding uninformative high-level ontology terms (like BFO:0000001 "entity") while keeping domain-specific parent classes (like ENVO:00010483 "environmental material").

```bash
# With default BFO exclusions (recommended - excludes 10 high-level BFO terms)
uv run mixs-triad-analysis parents \
  --file data/gold_flattened_biosamples_for_env_embeddings_202510061108.tsv \
  --source gold \
  --column env_medium

# Without any exclusions (show all parent classes including BFO)
uv run mixs-triad-analysis parents \
  --file data/gold_flattened_biosamples_for_env_embeddings_202510061108.tsv \
  --source gold \
  --column env_medium \
  --no-default-exclusions

# With custom exclusions (in addition to defaults)
# Example: also exclude astronomical body part and system
uv run mixs-triad-analysis parents \
  --file data/ncbi_flattened_biosamples_for_env_embeddings_202510061108_normalized.tsv \
  --source ncbi \
  --exclude "ENVO:01000813,RO:0002577"
```

**Default excluded terms** (BFO upper ontology classes):
- BFO:0000001 (entity)
- BFO:0000002 (continuant)
- BFO:0000003 (occurrent)
- BFO:0000004 (independent continuant)
- BFO:0000016 (disposition)
- BFO:0000017 (realizable entity)
- BFO:0000020 (specifically dependent continuant)
- BFO:0000024 (fiat object part)
- BFO:0000031 (generically dependent continuant)
- BFO:0000040 (material entity)

### Via justfile (simpler - uses defaults)
```bash
# Single source with custom threshold
just mixs-triad-frequency source=gold min_pct=0.2

# Parent analysis for specific MIxS triad column
just mixs-triad-parents source=gold column=env_medium

# All sources (uses default file paths and BFO exclusions)
just mixs-triad-all min_pct=0.1
```

## Recommended Thresholds for ML

Based on dataset characteristics:

- **0.01% (very permissive)**: Keep most terms, good for initial exploration
- **0.1% (default)**: Reasonable balance, filters out extremely rare terms
- **0.5% (stricter)**: For balanced training sets, removes rare classes
- **1.0% (very strict)**: Only common terms, best for highly imbalanced scenarios

For NCBI (452K records):
- 0.01% = 45 occurrences minimum
- 0.1% = 452 occurrences minimum
- 0.5% = 2,260 occurrences minimum
- 1.0% = 4,520 occurrences minimum

## Example Output Interpretation

```
Class Imbalance Analysis
Terms accounting for 50% of occurrences: 6 (2.5% of unique terms)
Terms accounting for 80% of occurrences: 15 (6.3% of unique terms)
Terms accounting for 95% of occurrences: 42 (17.5% of unique terms)
```

This means:
- Dataset is highly imbalanced
- Just 6 terms (2.5%) account for half of all data
- 82.5% of unique terms only account for 5% of occurrences
- Consider higher threshold (0.5% or 1.0%) to focus on well-represented classes

## Migration Notes

If you have existing scripts using the old API:

```python
# Old code
result = analyze_term_frequency(df, column, min_frequency=5)

# New code
result_df, stats = analyze_term_frequency(df, column, min_pct=0.1)
# Or keep similar behavior with absolute override
result_df, stats = analyze_term_frequency(df, column, min_pct=0.0, min_absolute=5)
```
