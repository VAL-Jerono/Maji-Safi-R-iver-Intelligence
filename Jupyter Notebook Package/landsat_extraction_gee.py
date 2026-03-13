"""
LANDSAT EXTRACTION USING GOOGLE EARTH ENGINE
- More reliable than Planetary Computer
- Better for bulk extraction
- Requires Google Earth Engine authentication (free account)

Setup:
1. pip install earthengine-api geemap
2. earthengine authenticate
3. Run this script
"""

import ee
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os
import json
from datetime import datetime, timedelta

# Initialize Earth Engine
try:
    ee.Initialize()
    print("✅ Google Earth Engine initialized")
except:
    print("""
❌ Google Earth Engine not authenticated!

Fix:
1. Install: pip install earthengine-api
2. Authenticate: earthengine authenticate
3. Run this script again
    """)
    exit()

def get_landsat_from_gee(lat, lon, date_str, buffer_m=50):
    """Extract Landsat 8/9 bands using Google Earth Engine."""
    try:
        date = pd.to_datetime(date_str)
        
        # Create point
        point = ee.Geometry.Point([lon, lat])
        
        # Create buffer (convert meters to degrees)
        buffer_degrees = buffer_m / 111_000
        roi = point.buffer(buffer_degrees)
        
        # Landsat 8 & 9 collections
        ls8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
            .filterBounds(roi) \
            .filterDate(ee.Date(date.strftime('%Y-%m-%d')).advance(-15, 'day'),
                       ee.Date(date.strftime('%Y-%m-%d')).advance(15, 'day')) \
            .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
            .sort('CLOUD_COVER')
        
        ls9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") \
            .filterBounds(roi) \
            .filterDate(ee.Date(date.strftime('%Y-%m-%d')).advance(-15, 'day'),
                       ee.Date(date.strftime('%Y-%m-%d')).advance(15, 'day')) \
            .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
            .sort('CLOUD_COVER')
        
        # Merge and get first image
        combined = ls8.merge(ls9)
        
        if combined.size().getInfo() == 0:
            return {
                "nir": np.nan, "green": np.nan, "swir16": np.nan, "swir22": np.nan,
                "error": "no_images"
            }
        
        image = ee.Image(combined.first())
        
        # Band names (Landsat C2 L2)
        # SR_B2: Blue, SR_B3: Green, SR_B4: Red, SR_B5: NIR, SR_B6: SWIR1, SR_B7: SWIR2
        bands = {
            "nir": "SR_B5",      # NIR
            "green": "SR_B3",    # Green
            "swir16": "SR_B6",   # SWIR1 (1600nm)
            "swir22": "SR_B7"    # SWIR2 (2200nm)
        }
        
        results = {}
        for b_name, b_key in bands.items():
            try:
                # Get median value in region
                stat = image.select(b_key).reduceRegion(
                    reducer=ee.Reducer.median(),
                    geometry=roi,
                    scale=30  # 30m resolution
                )
                
                val = stat.getInfo().get(b_key)
                results[b_name] = float(val) if val is not None else np.nan
            except:
                results[b_name] = np.nan
        
        results["error"] = None
        return results
        
    except Exception as e:
        return {
            "nir": np.nan, "green": np.nan, "swir16": np.nan, "swir22": np.nan,
            "error": str(e)[:50]
        }

def add_indices(df):
    """Add vegetation indices."""
    df = df.copy()
    eps = 1e-10
    df['NDMI'] = (df['nir'] - df['swir16']) / (df['nir'] + df['swir16'] + eps)
    df['MNDWI'] = (df['green'] - df['swir16']) / (df['green'] + df['swir16'] + eps)
    return df

def load_checkpoint(buffer_m):
    """Load checkpoint."""
    checkpoint_file = f"checkpoint_gee_{buffer_m}m.json"
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return {
        "buffer_m": int(buffer_m),
        "last_completed_row": -1,
        "total_rows_processed": 0,
        "total_rows_with_data": 0,
        "start_time": datetime.now().isoformat(),
        "batches": {}
    }

def save_checkpoint(buffer_m, checkpoint):
    """Save checkpoint."""
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
    checkpoint_file = f"checkpoint_gee_{buffer_m}m.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_clean, f, indent=2)

def extract_buffer_gee(buffer_m, train_locs, batch_size=10):
    """Extract using Google Earth Engine."""
    
    print(f"\n{'='*80}")
    print(f"🎯 EXTRACTING {buffer_m}m BUFFER (Google Earth Engine)")
    print(f"{'='*80}")
    
    checkpoint = load_checkpoint(buffer_m)
    last_completed_row = checkpoint['last_completed_row']
    
    if last_completed_row >= 0:
        print(f"\n✅ RESUMING from row {last_completed_row + 1}")
        print(f"   Progress: {checkpoint['total_rows_processed']}/{len(train_locs)}")
    else:
        print(f"\n🆕 STARTING FRESH")
        print(f"   Total rows: {len(train_locs):,}")
        print(f"   Batch size: {batch_size} rows")
        print(f"   Note: GEE is slower but more reliable (~1-2 min per row)")
    
    total_with_data = checkpoint['total_rows_with_data']
    total_processed = checkpoint['total_rows_processed']
    batch_times = []
    
    num_batches = (len(train_locs) + batch_size - 1) // batch_size
    
    for batch_num in range(num_batches):
        batch_start_row = batch_num * batch_size
        batch_end_row = min(batch_start_row + batch_size, len(train_locs))
        
        if batch_end_row - 1 <= last_completed_row:
            continue
        
        batch_time_start = time.time()
        pct_complete = 100 * total_processed / len(train_locs)
        
        print(f"\n📦 BATCH {batch_num + 1}/{num_batches} | {batch_start_row:04d}-{batch_end_row:04d} | {pct_complete:.1f}%")
        
        batch_locs = train_locs.iloc[batch_start_row:batch_end_row].reset_index(drop=True)
        batch_features = []
        batch_errors = {}
        
        for idx, (_, row) in enumerate(tqdm(batch_locs.iterrows(), total=len(batch_locs),
                                             desc=f"  GEE Extract", leave=False, ncols=80)):
            bands = get_landsat_from_gee(row['Latitude'], row['Longitude'], 
                                        row['Sample Date'], buffer_m=buffer_m)
            
            bands['Latitude'] = float(row['Latitude'])
            bands['Longitude'] = float(row['Longitude'])
            bands['Sample Date'] = str(row['Sample Date'])
            bands['YearMonth'] = str(row['YearMonth'])
            
            batch_features.append(bands)
            
            if bands['error']:
                error = bands['error']
                batch_errors[error] = batch_errors.get(error, 0) + 1
        
        batch_df = pd.DataFrame(batch_features)
        batch_df = add_indices(batch_df)
        
        batch_with_data = batch_df['nir'].notna().sum()
        total_with_data += batch_with_data
        total_processed += len(batch_df)
        
        batch_time_end = time.time()
        batch_duration = batch_time_end - batch_time_start
        batch_times.append(batch_duration)
        
        pct = 100 * total_processed / len(train_locs)
        success_rate = 100 * total_with_data / total_processed if total_processed > 0 else 0
        
        print(f"   ✅ {batch_with_data}/{len(batch_df)} with data")
        print(f"   📊 Total: {total_with_data}/{total_processed} ({success_rate:.1f}%)")
        print(f"   ⏱️ {pct:.1f}% | Speed: {batch_duration:.0f}s")
        
        if batch_errors:
            print(f"   ⚠️ Errors: {dict(batch_errors)}")
        
        batch_filename = f"landsat_gee_{buffer_m}m_batch_{batch_start_row:04d}_{batch_end_row:04d}.csv"
        batch_df.to_csv(batch_filename, index=False)
        
        checkpoint['last_completed_row'] = int(batch_end_row - 1)
        checkpoint['total_rows_processed'] = int(total_processed)
        checkpoint['total_rows_with_data'] = int(total_with_data)
        checkpoint['batches'][f"batch_{batch_start_row:04d}_{batch_end_row:04d}"] = {
            "rows": int(len(batch_df)),
            "with_data": int(batch_with_data),
            "coverage": float(100 * batch_with_data / len(batch_df)) if len(batch_df) > 0 else 0.0
        }
        
        save_checkpoint(buffer_m, checkpoint)
        time.sleep(1)
    
    # Combine
    print(f"\n{'='*80}")
    print(f"🔗 COMBINING {buffer_m}m BATCHES")
    print(f"{'='*80}\n")
    
    all_batches = []
    batch_files = sorted([f for f in os.listdir('.') if f.startswith(f"landsat_gee_{buffer_m}m_batch_")])
    
    for batch_file in batch_files:
        try:
            df = pd.read_csv(batch_file)
            all_batches.append(df)
            print(f"  ✅ {batch_file}")
        except Exception as e:
            print(f"  ⚠️ {batch_file}: {e}")
    
    if all_batches:
        final_df = pd.concat(all_batches, ignore_index=True)
        final_filename = f"landsat_gee_{buffer_m}m_COMPLETE.csv"
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
    print("🚀 LANDSAT EXTRACTION - GOOGLE EARTH ENGINE")
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
        result = extract_buffer_gee(buffer_m, train_locs, batch_size=10)
        if result is not None:
            results[buffer_m] = result
    
    print("\n\n" + "="*80)
    print("✅ ALL COMPLETE!")
    print("="*80)
    
    if results:
        print("\n📁 Files:")
        for buffer_m in [50, 150, 200]:
            fn = f"landsat_gee_{buffer_m}m_COMPLETE.csv"
            if os.path.exists(fn):
                print(f"   ✅ {fn}")
        
        print("\n📊 Summary:")
        for buffer_m, df in results.items():
            wd = df['nir'].notna().sum()
            pct = 100 * wd / len(df)
            print(f"   {buffer_m}m: {len(df):,} | {wd:,} with data ({pct:.1f}%)")
    
    print("\n✅ Ready for next step: Join with water quality data")