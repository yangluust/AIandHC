# Model Result Caching System

## Overview

The caching system allows you to save and reuse model results, eliminating the need to re-solve computationally expensive models for repeated analysis.

## Quick Start

### 1. Generate and Cache Results
```bash
# Run analysis and save results to cache
python quick_comparison.py --force-recompute

# This will:
# - Solve both full model and no-HC model
# - Save results to cache automatically
# - Display comprehensive analysis
```

### 2. Use Cached Results (Instant Analysis)
```bash
# Load and analyze cached results (very fast!)
python cached_analysis.py

# This will:
# - Load pre-computed results from cache
# - Display analysis and visualizations
# - Complete in seconds instead of minutes
```

## Command-Line Options

### quick_comparison.py
```bash
python quick_comparison.py                    # Use cache if available, save if computed
python quick_comparison.py --no-cache         # Don't use cache, always compute
python quick_comparison.py --no-save          # Don't save results to cache
python quick_comparison.py --force-recompute  # Force recomputation even if cached
```

### cached_analysis.py
```bash
python cached_analysis.py                     # Load and analyze cached results
python cached_analysis.py --no-plots          # Skip visualizations
python cached_analysis.py --list              # List available cached results
```

### Cache Management
```bash
# Interactive cache management
python utils/cache_manager.py

# Command-line cache management
python utils/cache_manager.py --list          # List cached results
python utils/cache_manager.py --stats         # Show cache statistics
python utils/cache_manager.py --clear         # Interactive cache clearing
```

## File Structure

```
code/
├── results/
│   ├── cache/
│   │   ├── full_model_[hash].pkl.gz         # Compressed model results
│   │   ├── no_hc_model_[hash].pkl.gz
│   │   └── ...
│   ├── metadata.json                        # Cache metadata
│   └── cache_index.json                     # Quick lookup index
├── utils/
│   ├── result_cache.py                      # Core caching utilities
│   └── cache_manager.py                     # Cache management tools
└── cached_analysis.py                       # Quick result viewer
```

## Key Features

### ✅ **Speed**
- Instant access to pre-computed results
- Analysis completes in seconds instead of minutes

### ✅ **Parameter Verification**
- Automatic parameter consistency checking
- Prevents loading incompatible cached results

### ✅ **Compression**
- Automatic compression of large result files
- Efficient disk space usage

### ✅ **Metadata Tracking**
- Timestamps, file sizes, parameter sets
- Easy identification of cached results

### ✅ **Backward Compatibility**
- All existing scripts work without modification
- Caching is optional and transparent

## Typical Workflow

### First Time (Generate Cache)
```bash
# 1. Run full analysis and cache results (takes time)
python quick_comparison.py --force-recompute

# 2. Verify cache was created
python utils/cache_manager.py --list
```

### Subsequent Analysis (Use Cache)
```bash
# 1. Quick analysis using cached results (instant)
python cached_analysis.py

# 2. Or run full analysis with cache (also instant)
python quick_comparison.py
```

### Cache Management
```bash
# View cache statistics
python utils/cache_manager.py --stats

# Clear old results
python utils/cache_manager.py --clear
```

## Cache Keys

Cache keys are generated from model parameters using MD5 hashing:
- Same parameters → Same cache key → Reuse results
- Different parameters → Different cache key → New computation

## Error Handling

The system gracefully handles:
- Missing cache files (falls back to computation)
- Parameter mismatches (warns and recomputes)
- Corrupted cache files (skips and recomputes)
- Missing dependencies (disables caching)

## Benefits

1. **Development Speed**: Instant iteration on analysis code
2. **Reproducibility**: Consistent results across sessions  
3. **Efficiency**: No redundant model solving
4. **Flexibility**: Easy parameter comparison
5. **Storage**: Organized result management

## Example Usage

```python
# In your own scripts
from utils.result_cache import get_cache, create_parameter_dict

# Define parameters
params = create_parameter_dict(
    Kai_n=0.195, Kai_e=0.123, e_l=1.0, e_h=2.0,
    lambda_param=0.4, r=0, w=1, discount=0.95
)

# Try to load from cache
cache = get_cache()
results = cache.load_model_results('full_model', params)

if results is None:
    # Compute and save
    results = run_my_model(params)
    cache.save_model_results(results, 'full_model', params)

# Use results...
```

## Cache Statistics

The system tracks:
- Number of cached files
- Total disk usage
- Results by model type
- Creation timestamps
- Parameter sets

Access via: `python utils/cache_manager.py --stats`
