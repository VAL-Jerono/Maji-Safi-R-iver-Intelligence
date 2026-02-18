# Local Setup Guide

## The Problem

Your notebooks were designed to run in **Snowflake's native notebook environment**, which supports:
- Native SQL cells (e.g., `show integrations;`)
- `get_active_session()` for automatic Snowflake connection
- Snowflake's container runtime with pre-installed packages

In a **local Jupyter/VS Code environment**, you need Python-only cells and explicit Snowflake connections.

## What Needs to Change?

### ❌ Won't Work Locally (Original Notebooks)

1. **SQL cells** - Lines like `show integrations;` cause `SyntaxError`
2. **`get_active_session()`** - This only exists in Snowflake's environment
3. **Snowflake file operations** - `PUT file://` commands for notebook stages
4. **`!uv pip install`** - Snowflake-specific package manager

### ✅ Already Working Locally

- All Python code for satellite data (pystac, xarray, matplotlib, etc.)
- Your utility scripts ([config.py](config.py), [src/snowflake_utils.py](src/snowflake_utils.py))
- Data processing and visualization code
- Machine learning code

## Quick Setup (3 Steps)

### 1. Install Dependencies (~5 minutes)

```bash
pip install -r requirements.txt
```

**Note:** Some packages like `rasterio` and `geopandas` have system dependencies. If you encounter errors:

**macOS:**
```bash
brew install gdal
pip install -r requirements.txt
```

**Linux:**
```bash
sudo apt-get install gdal-bin libgdal-dev
pip install -r requirements.txt
```

### 2. Configure Snowflake Credentials (~2 minutes)

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your credentials
# Use a text editor or:
nano .env
```

Fill in:
```
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
```

### 3. Use the Local Notebook

Open [notebooks/GETTING_STARTED_LOCAL.ipynb](notebooks/GETTING_STARTED_LOCAL.ipynb) - this is already adapted for local execution!

## What Changed in the Local Version?

### Original (Snowflake) → Local (Jupyter)

| Original | Local Alternative |
|----------|-------------------|
| `show integrations;` | ❌ Removed (not needed locally) |
| `session = get_active_session()` | `from src.snowflake_utils import SnowflakeManager`<br>`session = SnowflakeManager.connect()` |
| SQL cells | Wrapped in Python: `session.sql("SELECT ...").to_pandas()` |
| `PUT file://` | Not needed - use local file system directly |
| `!uv pip install` | Already installed via requirements.txt |

### What Stayed the Same?

✅ **ALL** the actual data processing code:
- Planetary Computer API calls
- Satellite data loading with `stac_load()`
- Cloud masking functions
- NDVI/NDMI calculations
- All visualizations with matplotlib

**About 90% of the code is unchanged!**

## Testing Your Setup

Run this to verify everything works:

```bash
python tests/test_connection.py
```

Expected output:
```
✅ Connected as: YOUR_USERNAME
✅ Using warehouse: COMPUTE_WH
✅ Quality data shape: (1000, X)
...
```

## Using Other Notebooks

Your other notebooks ([LANDSAT_DEMONSTRATION_NOTEBOOK_SNOWFLAKE.ipynb](notebooks/LANDSAT_DEMONSTRATION_NOTEBOOK_SNOWFLAKE.ipynb), etc.) need the same adaptations:

### Quick Fix for Any Notebook:

1. **Remove SQL-only cells** (lines like `show integrations;`)

2. **Replace Snowflake session code:**
   ```python
   # OLD (Snowflake):
   from snowflake.snowpark.context import get_active_session
   session = get_active_session()
   
   # NEW (Local):
   import sys
   sys.path.insert(0, '..')
   from src.snowflake_utils import SnowflakeManager
   session = SnowflakeManager.connect()
   ```

3. **Keep everything else the same!**

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'snowflake'`
**Fix:** Run `pip install -r requirements.txt`

### Issue: `KeyError: 'SNOWFLAKE_USER'`
**Fix:** Create `.env` file from `.env.example` with your credentials

### Issue: `rasterio` or `gdal` errors
**Fix:** Install system dependencies:
- macOS: `brew install gdal`
- Linux: `sudo apt-get install gdal-bin libgdal-dev`

### Issue: "Connection failed"
**Fix:** Check your `.env` credentials and network connection to Snowflake

## Summary

**Minimal changes needed:**
- ✅ Credentials in `.env` file
- ✅ Install packages once
- ✅ Use local notebook version or do simple find/replace in others

**Everything else works as-is:**
- ✅ All data processing
- ✅ All visualizations  
- ✅ All machine learning code
- ✅ All utility functions

You're ready to go! 🚀
