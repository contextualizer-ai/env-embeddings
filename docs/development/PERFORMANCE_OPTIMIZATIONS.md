# Performance Optimizations

## Problem: Slow Google Earth Engine Failures

When processing large datasets, failures were taking a long time because:
1. Each failure required a network request to Google Earth Engine
2. Fallback year attempts doubled the network calls
3. Repeated failures for the same location/year kept making the same slow requests

## Solution: Cached Failures + Verbose Logging

### 1. Cache Failures (Fast-Fail)

**Before:**
- Ocean location at (34.24, 144.31, 2010) → slow API call → failure
- Same location at (34.24, 144.31, 2010) → slow API call again → failure again

**After:**
- Ocean location at (34.24, 144.31, 2010) → slow API call → failure → **cached as None**
- Same location at (34.24, 144.31, 2010) → **instant cache lookup** → failure (cached)

**Implementation:**
```python
# earth_engine.py
if use_cache and cache_key in _cache:
    cached_value = _cache[cache_key]
    if cached_value is None:
        raise ValueError(f"No embedding found... (cached failure)")
    return cached_value

# Cache failures
except ValueError as e:
    if use_cache and "No embedding found" in str(e):
        _cache[cache_key] = None  # Cache the failure
    raise
```

### 2. Verbose Logging

**Before (with tqdm):**
```
Processing samples: 45%|███████████████              | 135/300 [02:15<02:45, 1.00row/s]
```
No visibility into:
- What coordinates are being processed
- Whether cache or API was used
- Why failures occurred

**After:**
```
Processing samples: 45%|███████████████              | 135/300 [02:15<02:45, 1.00row/s]
Row 132: ✓ Got embedding for (35.1180,138.9370) year=2017 [CACHE]
Row 133: ✗ No embedding for (34.2400,144.3100) year=1998 [API] - trying fallback year 2020
Row 133: ✗ No embedding for (34.2400,144.3100) year=2020 [API - FALLBACK] - SKIPPING
Row 134: ✓ Got embedding for (36.0540,140.1230) year=2015 [API]
Row 135: ✗ No embedding for (34.2400,144.3100) year=1998 [CACHED FAILURE] - trying fallback year 2020
Row 135: ✗ No embedding for (34.2400,144.3100) year=2020 [CACHED FAILURE - FALLBACK] - SKIPPING
```

Now you can see:
- **Coordinates**: Exact lat/lon being processed
- **Year**: Original year and fallback year
- **Source**: `[CACHE]` vs `[API]` - instant vs slow
- **Status**: ✓ success vs ✗ failure
- **Cached Failures**: Instant failure on second attempt

### 3. Performance Impact

**Scenario: 300 rows with 50% ocean locations**

**Before:**
- 150 ocean locations × 2 API calls (original + fallback) × 2 seconds = **600 seconds of API calls**
- Processing ocean samples multiple times → same slow failures

**After (first run):**
- 150 ocean locations × 2 API calls × 2 seconds = **600 seconds** (same on first run)
- But failures are cached

**After (second run or duplicate coordinates):**
- 150 ocean locations × instant cache lookup = **< 1 second**
- **600x speedup** for cached failures

### 4. Statistics Tracking

The `ProcessingStats` class now correctly tracks:
- **Cache hits**: Includes both successful cache hits AND cached failures
- **Cache misses**: Only new API calls
- **API success**: Successful new embeddings
- **API failures**: Categorized by type (no coverage, rate limit, other)

**Example output:**
```
=== Google Earth Embeddings Processing ===
Successfully retrieved embeddings: 245
  - From cache: 180
  - From API: 65

Failed to retrieve embeddings: 55
  - No coverage/ocean: 50
  - Rate limit (429): 0
  - Other errors: 5

Skipped rows:
  - Already had embeddings: 0
  - Invalid data (missing coords/dates): 0

Cache information:
  - Total entries in cache: 350 (includes 50 cached failures)
  - Cache directory: /Users/MAM/.cache/env-embeddings/google_earth
```

## Usage

### Commands

```bash
# Process 300 RANDOM rows - failures are now fast after first run
# Note: Uses random sampling (not sequential) to avoid bias
uv run env-embeddings add-google-embeddings-csv \
  data/satisfying_biosamples_normalized.csv \
  --max-rows 300 \
  --output data/temp_with_google_300.csv

uv run env-embeddings add-envo-embeddings-csv \
  data/temp_with_google_300.csv \
  --max-rows 300 \
  --output data/with_both_embeddings_300.csv
```

### Random Sampling (New in v2)

Both `add-google-embeddings-csv` and `add-envo-embeddings-csv` now use **random sampling** instead of sequential processing:

**Before:**
```python
df.head(max_rows)  # First N rows
```

**After:**
```python
df.sample(n=min(max_rows, len(df)), random_state=42)  # Random N rows
```

**Why this matters:**
- Avoids bias from data ordering (e.g., all samples from one region at start of file)
- Provides representative sample across geographic/temporal diversity
- Uses fixed random seed (42) for reproducibility
- Critical for scientific validity when sampling subsets

### What to Expect

**First run with new data:**
- Ocean/no-coverage locations will be slow (2 API calls each)
- But you'll see exactly which coordinates are failing and why
- Failures are cached for next time

**Subsequent runs or duplicate coordinates:**
- Cached failures are instant
- Progress bar moves much faster through known-bad locations
- Verbose output shows `[CACHED FAILURE]` for instant failures

### Debugging Failed Requests

With the new verbose logging, you can:

1. **Identify problem coordinates:**
   ```
   Row 133: ✗ No embedding for (34.2400,144.3100) year=1998 [API] - SKIPPING
   ```
   → This is an ocean location, no coverage

2. **See year issues:**
   ```
   Row 142: ✗ No embedding for (51.5074,-0.1278) year=1985 [API] - trying fallback year 2020
   Row 142: ✓ Got embedding for (51.5074,-0.1278) year=2020 [API - FALLBACK]
   ```
   → 1985 has no data, but 2020 works

3. **Spot patterns:**
   - Multiple failures with similar coordinates → might be ocean region
   - All failures in certain year range → dataset coverage issue
   - Specific error messages → can debug with Google Earth Engine docs

## Cache Management

### View Cache

```bash
# Google Earth cache
ls -lh ~/.cache/env-embeddings/google_earth/

# ENVO cache
ls -lh ~/.cache/env-embeddings/envo/
```

### Clear Cache (if needed)

```python
from env_embeddings.earth_engine import clear_cache
clear_cache()
```

Or manually:
```bash
rm -rf ~/.cache/env-embeddings/google_earth/
rm -rf ~/.cache/env-embeddings/envo/
```

## Benefits

✅ **Fast failures** - Cached failures are instant (600x faster)
✅ **Detailed logging** - See exactly what's happening with each request
✅ **Better debugging** - Coordinates, years, and error types visible
✅ **Cache visibility** - Know when cache vs API is used
✅ **Accurate statistics** - Proper tracking of cache hits, API calls, failures
✅ **No behavior change** - Same results, just faster and more visible
