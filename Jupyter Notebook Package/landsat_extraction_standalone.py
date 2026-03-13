"""
LANDSAT EXTRACTION - FINAL VERSION
- Fixes: JSON serialization of numpy types
- Better error handling
- Smaller batches (50 rows) to avoid timeouts
- Resume capability with checkpoints
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
import rioxarray
from tqdm import tqdm
import time
import os
import json
from datetime import datetime, timedelta

# ── CREATE ROBUST SESSION ──
def create_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

api_session = create_session()

def sign_url(href, retries=3):
    """Sign Planetary Computer URL with retries."""
    for attempt in range(retries):
        try:
            url = f"https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={href}"
            resp = api_session.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()["href"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise

def get_landsat_bands(lat, lon, date_str, buffer_m=50):
    """Extract Landsat bands for a single location."""
    
    try:
        date = pd.to_datetime(date_str)
        bbox_size = buffer_m / 111_000.0
        bbox = [lon - bbox_size, lat - bbox_size, lon + bbox_size, lat + bbox_size]
        
        # Search
        stac_api = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
        params = {
            "collections": ["landsat-c2-l2"],
            "bbox": bbox,
            "datetime": "2011-01-01/2015-12-31",
            "query": {"eo:cloud_cover": {"lt": 20}},
            "limit": 50
        }
        
        resp = api_session.post(stac_api, json=params, timeout=60)
        resp.raise_for_status()
        items = resp.json().get('features', [])
        
        if not items:
            return {"nir": np.nan, "green": np.nan, "swir16": np.nan, "swir22": np.nan, "error": "no_items"}
        
        # Get closest by date
        sample_dt = pd.Timestamp(date).tz_localize("UTC") if not date.tzinfo else date
        item = min(items, key=lambda x: abs(pd.to_datetime(x['properties']['datetime']).tz_convert("UTC") - sample_dt))
        
        # Load bands
        results = {}
        for b_name, b_key in [("nir", "nir08"), ("green", "green"), ("swir16", "swir16"), ("swir22", "swir22")]:
            try:
                href = item['assets'][b_key]['href']
                signed = sign_url(href)
                da = rioxarray.open_rasterio(signed, masked=True).rio.clip_box(*bbox)
                val = da.median(skipna=True).values.item()
                results[b_name] = float(val) if not np.isnan(val) else np.nan
            except Exception as e:
                results[b_name] = np.nan
        
        results["error"] = None
        return results
        
    except Exception as e:
        error = str(e)[:50]
        return {"nir": np.nan, "green": np.nan, "swir16": np.nan, "swir22": np.nan, "error": error}

def add_indices(df):
    """Add vegetation indices."""
    df = df.copy()
    eps = 1e-10
    df['NDMI'] = (df['nir'] - df['swir16']) / (df['nir'] + df['swir16'] + eps)
    df['MNDWI'] = (df['green'] - df['swir16']) / (df['green'] + df['swir16'] + eps)
    return df

def load_checkpoint(buffer_m):
    """Load checkpoint to see where we left off."""
    checkpoint_file = f"checkpoint_{buffer_m}m.json"
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            return checkpoint
        except:
            pass
    
    return {
        "buffer_m": int(buffer_m),
        "last_completed_batch": -1,
        "last_completed_row": -1,
        "total_rows_processed": 0,
        "total_rows_with_data": 0,
        "start_time": datetime.now().isoformat(),
        "batches": {}
    }

def save_checkpoint(buffer_m, checkpoint):
    """Save checkpoint after each batch - CONVERT NUMPY TYPES TO NATIVE PYTHON."""
    # Convert ALL numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    checkpoint_clean = convert_types(checkpoint)
    
    checkpoint_file = f"checkpoint_{buffer_m}m.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_clean, f, indent=2)

def extract_buffer(buffer_m, train_locs, batch_size=50):
    """Extract Landsat data for a specific buffer size."""
    
    print(f"\n{'='*80}")
    print(f"🎯 EXTRACTING {buffer_m}m BUFFER")
    print(f"{'='*80}")
    
    checkpoint = load_checkpoint(buffer_m)
    last_completed_row = checkpoint['last_completed_row']
    
    if last_completed_row >= 0:
        print(f"\n✅ RESUMING from row {last_completed_row + 1}")
        print(f"   Progress: {checkpoint['total_rows_processed']}/{len(train_locs)} rows")
        pct = 100 * checkpoint['total_rows_processed'] / len(train_locs)
        print(f"   Completion: {pct:.1f}%")
        if checkpoint['total_rows_processed'] > 0:
            success_rate = 100 * checkpoint['total_rows_with_data'] / checkpoint['total_rows_processed']
            print(f"   Success rate: {success_rate:.1f}%")
    else:
        print(f"\n🆕 STARTING FRESH")
        print(f"   Total rows: {len(train_locs):,}")
        print(f"   Batch size: {batch_size} rows")
        print(f"   Total batches: {(len(train_locs) + batch_size - 1) // batch_size}")
        print(f"   ⏱️ Est. time: {len(train_locs) * 12 / 3600:.1f} hours")
    
    total_with_data = checkpoint['total_rows_with_data']
    total_processed = checkpoint['total_rows_processed']
    batch_times = []
    
    num_batches = (len(train_locs) + batch_size - 1) // batch_size
    
    for batch_num in range(num_batches):
        batch_start_row = batch_num * batch_size
        batch_end_row = min(batch_start_row + batch_size, len(train_locs))
        
        # Skip if already completed
        if batch_end_row - 1 <= last_completed_row:
            continue
        
        batch_time_start = time.time()
        pct_complete = 100 * total_processed / len(train_locs)
        
        print(f"\n📦 BATCH {batch_num + 1}/{num_batches} | {batch_start_row:04d}-{batch_end_row:04d} | {pct_complete:.1f}%")
        
        batch_locs = train_locs.iloc[batch_start_row:batch_end_row].reset_index(drop=True)
        batch_features = []
        batch_errors = {}
        
        for idx, (_, row) in enumerate(tqdm(batch_locs.iterrows(), total=len(batch_locs), 
                                             desc=f"  Extracting", leave=False, ncols=80)):
            bands = get_landsat_bands(row['Latitude'], row['Longitude'], row['Sample Date'], buffer_m=buffer_m)
            
            bands['Latitude'] = float(row['Latitude'])
            bands['Longitude'] = float(row['Longitude'])
            bands['Sample Date'] = str(row['Sample Date'])
            bands['YearMonth'] = str(row['YearMonth'])
            
            batch_features.append(bands)
            
            if bands['error']:
                error = bands['error']
                batch_errors[error] = batch_errors.get(error, 0) + 1
            
            if (idx + 1) % 15 == 0:
                time.sleep(1)
        
        batch_df = pd.DataFrame(batch_features)
        batch_df = add_indices(batch_df)
        
        batch_with_data = batch_df['nir'].notna().sum()
        total_with_data += batch_with_data
        total_processed += len(batch_df)
        
        batch_time_end = time.time()
        batch_duration = batch_time_end - batch_time_start
        batch_times.append(batch_duration)
        
        if batch_times:
            avg_batch_time = np.mean(batch_times)
            remaining_batches = num_batches - batch_num - 1
            eta_seconds = remaining_batches * avg_batch_time
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)
            eta_str = eta_time.strftime("%H:%M")
        else:
            eta_str = "..."
        
        pct = 100 * total_processed / len(train_locs)
        success_rate = 100 * total_with_data / total_processed if total_processed > 0 else 0
        
        print(f"   ✅ {batch_with_data}/{len(batch_df)} with data")
        print(f"   📊 Total: {total_with_data}/{total_processed} ({success_rate:.1f}%)")
        print(f"   ⏱️ {pct:.1f}% | ETA: {eta_str} | Speed: {batch_duration:.0f}s")
        
        if batch_errors:
            print(f"   ⚠️ Errors: {dict(batch_errors)}")
        
        batch_filename = f"landsat_{buffer_m}m_batch_{batch_start_row:04d}_{batch_end_row:04d}.csv"
        batch_df.to_csv(batch_filename, index=False)
        
        # Update checkpoint - use int() and float() to ensure native types
        checkpoint['last_completed_batch'] = int(batch_num)
        checkpoint['last_completed_row'] = int(batch_end_row - 1)
        checkpoint['total_rows_processed'] = int(total_processed)
        checkpoint['total_rows_with_data'] = int(total_with_data)
        checkpoint['batches'][f"batch_{batch_start_row:04d}_{batch_end_row:04d}"] = {
            "rows": int(len(batch_df)),
            "with_data": int(batch_with_data),
            "coverage": float(100 * batch_with_data / len(batch_df)) if len(batch_df) > 0 else 0.0
        }
        
        save_checkpoint(buffer_m, checkpoint)
        time.sleep(2)
    
    # ── COMBINE BATCHES ──
    print(f"\n{'='*80}")
    print(f"🔗 COMBINING {buffer_m}m BATCHES")
    print(f"{'='*80}\n")
    
    all_batches = []
    batch_files = sorted([f for f in os.listdir('.') if f.startswith(f"landsat_{buffer_m}m_batch_")])
    
    for batch_file in batch_files:
        try:
            df = pd.read_csv(batch_file)
            all_batches.append(df)
            print(f"  ✅ {batch_file}")
        except Exception as e:
            print(f"  ⚠️ {batch_file}: {e}")
    
    if all_batches:
        final_df = pd.concat(all_batches, ignore_index=True)
        final_filename = f"landsat_{buffer_m}m_COMPLETE.csv"
        final_df.to_csv(final_filename, index=False)
        
        final_with_data = final_df['nir'].notna().sum()
        final_coverage = 100 * final_with_data / len(final_df)
        
        print(f"\n{'='*80}")
        print(f"✅ {buffer_m}m COMPLETE")
        print(f"{'='*80}")
        print(f"📁 {final_filename}")
        print(f"📊 Total: {len(final_df):,} rows")
        print(f"✅ With data: {final_with_data:,}")
        print(f"📈 Coverage: {final_coverage:.1f}%")
        
        if final_coverage > 0:
            print(f"\n📋 Sample:")
            print(final_df[['Latitude', 'Longitude', 'nir', 'green', 'NDMI']].head(3).to_string())
        
        return final_df
    
    return None

# ── MAIN ──
if __name__ == "__main__":
    print("="*80)
    print("🚀 LANDSAT EXTRACTION - FINAL")
    print("="*80)
    
    print("\n📂 Loading data...")
    df = pd.read_csv('water_quality_training_dataset.csv')
    df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    df['YearMonth'] = df['Sample Date'].dt.to_period('M')
    
    train_locs = df[['Latitude','Longitude','Sample Date','YearMonth']].drop_duplicates(
        subset=['Latitude','Longitude','YearMonth']
    ).reset_index(drop=True)
    
    print(f"✅ Total: {len(train_locs):,} location-month pairs\n")
    
    results = {}
    for buffer_m in [50, 150, 200]:
        result = extract_buffer(buffer_m, train_locs, batch_size=50)
        if result is not None:
            results[buffer_m] = result
    
    print("\n\n" + "="*80)
    print("✅ ALL COMPLETE!")
    print("="*80)
    
    if results:
        print("\n📁 Files:")
        for buffer_m in [50, 150, 200]:
            fn = f"landsat_{buffer_m}m_COMPLETE.csv"
            if os.path.exists(fn):
                print(f"   ✅ {fn}")
        
        print("\n📊 Summary:")
        for buffer_m, df in results.items():
            wd = df['nir'].notna().sum()
            pct = 100 * wd / len(df)
            print(f"   {buffer_m}m: {len(df):,} | {wd:,} with data ({pct:.1f}%)")
    
    print("\n✅ Ready for next step: Join with water quality data")