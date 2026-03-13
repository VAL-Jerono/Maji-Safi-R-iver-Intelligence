"""
LANDSAT BATCHED EXTRACTION - FINAL WORKING VERSION
==================================================

CRITICAL FIX:
- Problem: Landsat C2 L2 data is in UTM (EPSG:32634)
- Original code: Tried to clip with WGS84 lat/lon coordinates
- Result: "No data found in bounds" error
- Solution: Convert WGS84 bbox to UTM before clipping

The fix:
- Use pyproj Transformer to convert bbox from WGS84 to UTM
- Clip raster in its native coordinate system
- Extract actual band values!
"""

import warnings
warnings.filterwarnings('ignore')

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
import rioxarray
from pyproj import Transformer
from tqdm import tqdm
import time
import os
import json
from datetime import datetime, timedelta
import glob
import sys
import math

# ════════════════════════════════════════════════════════════════
# SESSION & API SETUP
# ════════════════════════════════════════════════════════════════

def create_robust_session():
    """Create session with connection pooling and retry logic."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=5,
        pool_maxsize=5
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

api_session = create_robust_session()
tile_cache = {}

# Create transformer for coordinate conversion (WGS84 -> UTM Zone 34S)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)

def sign_url(href, retries=2):
    """Sign Planetary Computer URL with exponential backoff."""
    for attempt in range(retries):
        try:
            url = f"https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={href}"
            resp = api_session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()["href"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None

# ════════════════════════════════════════════════════════════════
# TILE SEARCH & CACHING
# ════════════════════════════════════════════════════════════════

def get_landsat_tiles(bbox, date_range, cache=True):
    """Get Landsat items for a bbox (in WGS84)."""
    bbox_key = f"{bbox[0]:.3f}_{bbox[1]:.3f}_{bbox[2]:.3f}_{bbox[3]:.3f}"
    
    if cache and bbox_key in tile_cache:
        return tile_cache[bbox_key]
    
    try:
        stac_api = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
        params = {
            "collections": ["landsat-c2-l2"],
            "bbox": bbox,
            "datetime": date_range,
            "query": {"eo:cloud_cover": {"lt": 30}},
            "limit": 10
        }
        
        resp = api_session.post(stac_api, json=params, timeout=30)
        resp.raise_for_status()
        items = resp.json().get('features', [])
        
        if cache:
            tile_cache[bbox_key] = items
        return items
        
    except Exception as e:
        return []

# ════════════════════════════════════════════════════════════════
# BAND EXTRACTION - FIXED WITH COORDINATE CONVERSION
# ════════════════════════════════════════════════════════════════

def extract_bands(item, lat, lon, buffer_m):
    """
    Extract Landsat bands for a location.
    
    CRITICAL FIX: Convert WGS84 bbox to UTM before clipping!
    Landsat C2 L2 data is in EPSG:32634 (UTM Zone 34S)
    """
    try:
        # Create bounding box in WGS84 (lat/lon)
        bbox_size_deg = buffer_m / 111_000.0  # Simple approximation
        bbox_wgs84 = [lon - bbox_size_deg, lat - bbox_size_deg, 
                      lon + bbox_size_deg, lat + bbox_size_deg]
        
        # Convert bbox corners from WGS84 to UTM
        lon_min, lat_min, lon_max, lat_max = bbox_wgs84
        x_min, y_min = transformer.transform(lon_min, lat_min)
        x_max, y_max = transformer.transform(lon_max, lat_max)
        
        # Create bbox in UTM coordinates
        bbox_utm = [min(x_min, x_max), min(y_min, y_max), 
                    max(x_min, x_max), max(y_min, y_max)]
        
        results = {}
        for b_name, b_key in [("nir", "nir08"), ("green", "green"), 
                               ("swir16", "swir16"), ("swir22", "swir22")]:
            try:
                if b_key not in item['assets']:
                    results[b_name] = np.nan
                    continue
                
                href = item['assets'][b_key]['href']
                signed = sign_url(href)
                
                if not signed:
                    results[b_name] = np.nan
                    continue
                
                # Open raster (in UTM coordinates)
                da = rioxarray.open_rasterio(signed, masked=True)
                
                # Clip using UTM bbox (not WGS84!)
                da_clipped = da.rio.clip_box(*bbox_utm)
                
                # Extract median value
                val = da_clipped.median(skipna=True).values
                if val.ndim > 0:
                    val = val.item()
                
                results[b_name] = float(val) if not np.isnan(val) else np.nan
                
            except Exception as e:
                # If extraction fails, return NaN
                results[b_name] = np.nan
        
        return results
        
    except Exception as e:
        return {"nir": np.nan, "green": np.nan, "swir16": np.nan, "swir22": np.nan}

# ════════════════════════════════════════════════════════════════
# INDICES & FEATURES
# ════════════════════════════════════════════════════════════════

def compute_indices(row):
    """Compute NDMI and MNDWI from band reflectance values."""
    eps = 1e-10
    
    nir = row['nir']
    green = row['green']
    swir16 = row['swir16']
    swir22 = row['swir22']
    
    if pd.isna(nir) or pd.isna(swir16):
        row['NDMI'] = np.nan
    else:
        row['NDMI'] = (nir - swir16) / (nir + swir16 + eps)
    
    if pd.isna(green) or pd.isna(swir16):
        row['MNDWI'] = np.nan
    else:
        row['MNDWI'] = (green - swir16) / (green + swir16 + eps)
    
    return row

# ════════════════════════════════════════════════════════════════
# SPATIAL BINNING
# ════════════════════════════════════════════════════════════════

def spatial_bin_locations(locs, grid_size_km=10):
    """Group locations by spatial grid."""
    locs = locs.copy()
    locs['grid_x'] = (locs['Longitude'] * 111 / grid_size_km).astype(int)
    locs['grid_y'] = (locs['Latitude'] * 111 / grid_size_km).astype(int)
    return locs

# ════════════════════════════════════════════════════════════════
# BATCH MANAGEMENT
# ════════════════════════════════════════════════════════════════

def find_existing_batches(buffer_m):
    """Find all completed batch files for this buffer."""
    pattern = f"landsat_{buffer_m}m_batch_*.csv"
    files = sorted(glob.glob(pattern))
    
    batches = []
    for f in files:
        try:
            batch_num = int(f.split('_batch_')[1].split('.')[0])
            batches.append((batch_num, f))
        except:
            pass
    
    return batches

def get_next_batch_number(buffer_m):
    """Determine which batch to start from."""
    batches = find_existing_batches(buffer_m)
    
    if not batches:
        return 0
    
    last_batch_num = batches[-1][0]
    return last_batch_num + 1

def combine_batches(buffer_m):
    """Combine all batch files into final output."""
    batches = find_existing_batches(buffer_m)
    
    if not batches:
        print(f"❌ No batch files found for {buffer_m}m buffer")
        return None
    
    print(f"\n{'='*80}")
    print(f"🔗 COMBINING {len(batches)} BATCHES ({buffer_m}m)")
    print(f"{'='*80}\n")
    
    all_dfs = []
    total_rows = 0
    
    for batch_num, batch_file in batches:
        try:
            df = pd.read_csv(batch_file)
            all_dfs.append(df)
            total_rows += len(df)
            print(f"  ✅ {batch_file:40s} ({len(df):4d} rows)")
        except Exception as e:
            print(f"  ⚠️ {batch_file:40s} - ERROR: {str(e)[:40]}")
    
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        final_filename = f"landsat_{buffer_m}m_COMPLETE.csv"
        final_df.to_csv(final_filename, index=False)
        
        with_data = final_df['nir'].notna().sum()
        coverage = 100 * with_data / len(final_df) if len(final_df) > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"✅ COMBINED {buffer_m}m")
        print(f"{'='*80}")
        print(f"📁 Output: {final_filename}")
        print(f"📊 Total rows: {len(final_df):,}")
        print(f"✅ With data: {with_data:,}")
        print(f"📈 Coverage: {coverage:.1f}%")
        
        if coverage > 0:
            print(f"\n📋 Sample:")
            print(final_df[['Latitude', 'Longitude', 'nir', 'green', 'NDMI', 'MNDWI']].head(3).to_string())
        
        return final_df
    
    return None

# ════════════════════════════════════════════════════════════════
# MAIN BATCHED EXTRACTION
# ════════════════════════════════════════════════════════════════

def extract_landsat_batched(buffer_m, train_locs, date_range, batch_size=100):
    """
    Extract Landsat data in batches, saving each immediately.
    """
    
    print(f"\n{'='*80}")
    print(f"🎯 EXTRACTING {buffer_m}m BUFFER (BATCHED) - FINAL FIXED")
    print(f"{'='*80}\n")
    
    print(f"📅 Date range: {date_range}")
    print(f"📍 Total locations: {len(train_locs):,}")
    print(f"📦 Batch size: {batch_size} locations")
    
    num_batches = (len(train_locs) + batch_size - 1) // batch_size
    print(f"📊 Total batches: {num_batches}\n")
    
    start_batch = get_next_batch_number(buffer_m)
    existing_batches = find_existing_batches(buffer_m)
    
    if existing_batches:
        print(f"✅ RESUMING: Found {len(existing_batches)} completed batches")
        print(f"   Starting from batch {start_batch}\n")
    else:
        print(f"🆕 STARTING FRESH\n")
    
    locs_binned = spatial_bin_locations(train_locs, grid_size_km=10)
    
    overall_start_time = time.time()
    batch_times = []
    total_rows_processed = 0
    total_rows_with_data = 0
    
    # Process each batch
    for batch_num in range(start_batch, num_batches):
        batch_start_row = batch_num * batch_size
        batch_end_row = min(batch_start_row + batch_size, len(train_locs))
        
        batch_start_time = time.time()
        
        pct_complete = 100 * batch_start_row / len(train_locs)
        print(f"📦 BATCH {batch_num + 1}/{num_batches} | Rows {batch_start_row:04d}-{batch_end_row:04d} | {pct_complete:.1f}%")
        
        batch_locs = train_locs.iloc[batch_start_row:batch_end_row].reset_index(drop=True)
        batch_locs_binned = locs_binned.iloc[batch_start_row:batch_end_row].reset_index(drop=True)
        
        batch_features = []
        
        with tqdm(total=len(batch_locs), desc="  Extracting", ncols=80, leave=False) as pbar:
            for idx in range(len(batch_locs)):
                row = batch_locs.iloc[idx]
                row_binned = batch_locs_binned.iloc[idx]
                
                # Get tile for this grid cell
                lat_center = row_binned['Latitude']
                lon_center = row_binned['Longitude']
                
                tile_size = (buffer_m * 3) / 111_000.0
                tile_bbox = [
                    lon_center - tile_size,
                    lat_center - tile_size,
                    lon_center + tile_size,
                    lat_center + tile_size
                ]
                
                items = get_landsat_tiles(tile_bbox, date_range)
                
                if items:
                    try:
                        sample_dt = pd.Timestamp(row['Sample Date']).tz_localize("UTC")
                        closest_item = min(
                            items,
                            key=lambda x: abs(
                                pd.to_datetime(x['properties']['datetime']).tz_convert("UTC") - sample_dt
                            )
                        )
                        
                        bands = extract_bands(closest_item, row['Latitude'], row['Longitude'], buffer_m)
                        
                    except Exception as e:
                        bands = {"nir": np.nan, "green": np.nan, "swir16": np.nan, "swir22": np.nan}
                else:
                    bands = {"nir": np.nan, "green": np.nan, "swir16": np.nan, "swir22": np.nan}
                
                bands['Latitude'] = float(row['Latitude'])
                bands['Longitude'] = float(row['Longitude'])
                bands['Sample Date'] = str(row['Sample Date']).split()[0]
                bands['YearMonth'] = str(row['YearMonth'])
                
                batch_features.append(bands)
                
                pbar.update(1)
                
                if (idx + 1) % 10 == 0:
                    time.sleep(0.5)
        
        # Process batch dataframe
        batch_df = pd.DataFrame(batch_features)
        batch_df = batch_df.apply(compute_indices, axis=1)
        
        cols = ['Latitude', 'Longitude', 'Sample Date', 'YearMonth',
                'nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI']
        batch_df = batch_df[cols]
        
        # Stats
        batch_with_data = batch_df['nir'].notna().sum()
        total_rows_processed += len(batch_df)
        total_rows_with_data += batch_with_data
        
        batch_duration = time.time() - batch_start_time
        batch_times.append(batch_duration)
        
        if batch_times:
            avg_batch_time = np.mean(batch_times)
            remaining_batches = num_batches - batch_num - 1
            eta_seconds = remaining_batches * avg_batch_time
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)
            eta_str = eta_time.strftime("%H:%M")
        else:
            eta_str = "..."
        
        batch_coverage = 100 * batch_with_data / len(batch_df) if len(batch_df) > 0 else 0
        overall_coverage = 100 * total_rows_with_data / total_rows_processed if total_rows_processed > 0 else 0
        
        print(f"   ✅ {batch_with_data:4d}/{len(batch_df)} with data ({batch_coverage:5.1f}%)")
        print(f"   📊 Overall: {total_rows_with_data:,}/{total_rows_processed:,} ({overall_coverage:.1f}%)")
        print(f"   ⏱️ {batch_duration:6.0f}s | ETA: {eta_str}")
        
        batch_filename = f"landsat_{buffer_m}m_batch_{batch_start_row:04d}.csv"
        batch_df.to_csv(batch_filename, index=False)
        print(f"   💾 Saved: {batch_filename}\n")
        
        time.sleep(1)
    
    overall_elapsed = (time.time() - overall_start_time) / 60
    print(f"\n⏱️ Total extraction time: {overall_elapsed:.1f} minutes")
    
    final_df = combine_batches(buffer_m)
    
    return final_df

# ════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════

def main():
    print("="*80)
    print("🚀 LANDSAT BATCHED EXTRACTION - FINAL WORKING VERSION")
    print("="*80)
    print("Buffers: 50m, 150m, 200m (instructor recommended)")
    print("Fix: WGS84 -> UTM coordinate conversion before clipping\n")
    
    print("📂 Loading training data...")
    try:
        df = pd.read_csv('water_quality_training_dataset.csv')
    except FileNotFoundError:
        print("❌ File not found: water_quality_training_dataset.csv")
        sys.exit(1)
    
    df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    df['YearMonth'] = df['Sample Date'].dt.to_period('M')
    
    train_locs = df[['Latitude', 'Longitude', 'Sample Date', 'YearMonth']].drop_duplicates(
        subset=['Latitude', 'Longitude', 'YearMonth']
    ).reset_index(drop=True)
    
    date_min = train_locs['Sample Date'].min()
    date_max = train_locs['Sample Date'].max()
    date_range = f"{date_min.strftime('%Y-%m-%d')}/{date_max.strftime('%Y-%m-%d')}"
    
    print(f"✅ Loaded {len(df):,} total records")
    print(f"✅ Unique location-months: {len(train_locs):,}")
    print(f"📅 Date range detected: {date_range}")
    print(f"🌍 Geographic extent:")
    print(f"   Lat:  {train_locs['Latitude'].min():.2f}° to {train_locs['Latitude'].max():.2f}°")
    print(f"   Lon: {train_locs['Longitude'].min():.2f}° to {train_locs['Longitude'].max():.2f}°")
    
    # Extract multi-scale buffers
    results = {}
    for buffer_m in [50, 150, 200]:
        print(f"\n{'─'*80}")
        result = extract_landsat_batched(buffer_m, train_locs, date_range, batch_size=100)
        if result is not None:
            results[buffer_m] = result
        time.sleep(2)
    
    # Final summary
    print(f"\n\n{'='*80}")
    print(f"✅ EXTRACTION COMPLETE")
    print(f"{'='*80}\n")
    
    if results:
        print("📁 Generated COMPLETE files:")
        for buffer_m in [50, 150, 200]:
            fn = f"landsat_{buffer_m}m_COMPLETE.csv"
            if os.path.exists(fn):
                size = os.path.getsize(fn) / 1024 / 1024
                print(f"   ✅ {fn:35s} ({size:.1f} MB)")
        
        print("\n📊 Data coverage summary:")
        for buffer_m in [50, 150, 200]:
            if buffer_m in results:
                df_result = results[buffer_m]
                with_data = df_result['nir'].notna().sum()
                pct = 100 * with_data / len(df_result)
                print(f"   {buffer_m}m: {len(df_result):,} rows | {with_data:,} with data ({pct:.1f}%)")
        
        print("\n💡 Next steps:")
        print("   1. Check COMPLETE files have good coverage (40%+)")
        print("   2. Combine COMPLETE files if needed")
        print("   3. Join with water quality parameters")
        print("   4. Train predictive model")

if __name__ == "__main__":
    main()