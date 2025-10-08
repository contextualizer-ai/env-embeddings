# MIxS Triad Analysis - Renaming Summary

## What Changed

Renamed the analysis script and all references to reflect that this analyzes **MIxS environmental triad ontology terms**, not just ENVO terms specifically.

## MIxS Environmental Triad

The **MIxS (Minimum Information about any (x) Sequence) environmental triad** is a standard for describing environmental context using three hierarchical fields:

1. **env_broad_scale**: Broadest environmental context (typically a biome)
   - Example: `ENVO:00000446` (terrestrial biome)

2. **env_local_scale**: Local environmental context
   - Example: `ENVO:00000114` (agricultural field)

3. **env_medium**: Environmental material or medium
   - Example: `ENVO:00001998` (soil)

While these fields primarily use ENVO (Environment Ontology) terms, they can potentially include terms from other ontologies, hence the more accurate naming.

## Changes Made

### File Renamed
- `src/env_embeddings/envo_term_analysis.py` → `src/env_embeddings/mixs_triad_analysis.py`

### CLI Command Renamed
- **Old**: `uv run envo-analysis`
- **New**: `uv run mixs-triad-analysis`

### pyproject.toml
```toml
[project.scripts]
mixs-triad-analysis = "env_embeddings.mixs_triad_analysis:app"
```

### Justfile Recipes Renamed
- `envo-frequency` → `mixs-triad-frequency`
- `envo-parents` → `mixs-triad-parents`
- `envo-all` → `mixs-triad-all`
- Group: `envo-analysis` → `mixs-analysis`

### Code Changes
- `ENV_COLUMNS` → `MIXS_TRIAD_COLUMNS`
- `get_envo_adapter()` → `get_ontology_adapter()` (with note that ENVO is currently primary)
- Updated all docstrings to reference "MIxS triad" instead of "ENVO terms"
- Clarified that analysis is for "ontology terms in MIxS environmental triad fields"

### Documentation
- `docs/ENVO_ANALYSIS_UPDATES.md` → `docs/MIXS_TRIAD_ANALYSIS.md`
- Updated all usage examples with new command names

## Usage Examples

```bash
# CLI usage - Single source frequency analysis
uv run mixs-triad-analysis frequency \
  --file data/gold_flattened_biosamples_for_env_embeddings_202510061108.tsv \
  --source gold \
  --min-pct 0.1

# CLI usage - Parent analysis for specific column
uv run mixs-triad-analysis parents \
  --file data/ncbi_flattened_biosamples_for_env_embeddings_202510061108_normalized.tsv \
  --source ncbi \
  --column env_medium

# CLI usage - Analyze all three sources
uv run mixs-triad-analysis analyze-all \
  --gold data/gold_flattened_biosamples_for_env_embeddings_202510061108.tsv \
  --ncbi data/ncbi_flattened_biosamples_for_env_embeddings_202510061108_normalized.tsv \
  --nmdc data/nmdc_flattened_biosample_for_env_embeddings_202510061052.tsv \
  --min-pct 0.1

# Justfile usage (simpler - uses defaults)
just mixs-triad-frequency source=gold min_pct=0.2
just mixs-triad-parents source=ncbi column=env_medium
just mixs-triad-all min_pct=0.1
```

## Why This Naming is Better

1. **More Accurate**: Reflects that we're analyzing MIxS standard fields, not just ENVO
2. **Clearer Scope**: Makes it obvious we're analyzing the environmental triad specifically
3. **Extensible**: Allows for potential inclusion of non-ENVO ontology terms in the future
4. **Standards-Based**: References the actual MIxS standard that defines these fields

## Backward Compatibility

No backward compatibility maintained - this is a breaking change requiring:
- Update any scripts using old `envo-analysis` command to `mixs-triad-analysis`
- Update justfile calls from `envo-*` to `mixs-triad-*`
- Old output directory default was `results/envo_analysis`, new default is `results/mixs_triad_analysis`
