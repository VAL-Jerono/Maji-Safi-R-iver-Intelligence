# 🌊 Maji Safi River Intelligence
### EY AI & Data Challenge 2026 — Water Quality Prediction

> **Predicting river water quality across South Africa using satellite imagery, climate data, and machine learning.**

---

## 🏆 Results

| Metric | Value |
|--------|-------|
| **Best Leaderboard Score** | **0.3679** |
| Baseline Score | 0.1239 |
| Total Improvement | **+0.2440 (×2.97)** |
| Winning Optimization | Opt 11 — Temporal Lags |
| Total Experiments Run | 23+ |

---

## 📋 Challenge Overview

Predict three water quality parameters at ~200 river monitoring locations across South Africa (2011–2015), evaluated against **held-out spatial locations never seen during training**:

| Target | Description | Unit |
|--------|-------------|------|
| `Total Alkalinity` (TA) | Buffering capacity of water | mg/L |
| `Electrical Conductance` (EC) | Dissolved ion concentration | µS/cm |
| `Dissolved Reactive Phosphorus` (DRP) | Bioavailable phosphorus — nutrient pollution proxy | µg/L |

The core challenge: models are scored on **different river sites** from those in training. Any feature that encodes *where* a site is (coordinates, local thresholds) will memorise training locations and fail on unseen ones — **spatial overfitting** is the dominant failure mode.

---

## 🗄️ Data Sources & Infrastructure

### Snowflake — Data Mining Backbone

Snowflake served as the central data warehouse and query engine throughout the project. All feature extraction, dataset joining, temporal windowing, and quality control pipelines were built and run on Snowflake before delivering clean CSVs to the modelling environment.

```
Snowflake Roles:
├── Feature joins       → Joined training labels with TerraClimate + Landsat by Lat/Lon/Date
├── Spatial aggregation → Pre-computed buffer statistics at 50m, 100m, 150m, 200m
├── Temporal windowing  → Extracted month, season, day_of_year; partitioned by YearMonth
├── Quality control     → Identified 1,085 training rows with missing Landsat NIR band
└── Data delivery       → Exported row-aligned CSVs: 9,319 train + 200 validation rows
```

Without Snowflake's ability to join large satellite + climate datasets by spatial key and time window, the multi-source feature pipeline would have been orders of magnitude slower to iterate on.

### Satellite & Climate Data

| Source | Platform | Variables | Role |
|--------|----------|-----------|------|
| **Landsat 8** | Microsoft Planetary Computer | `nir`, `green`, `swir16`, `swir22`, `NDMI`, `MNDWI` at 100m buffer | Primary spectral signal — water clarity, turbidity, riparian vegetation |
| **TerraClimate** | Microsoft Planetary Computer | `pet`, `ppt`, `soil`, `tmax`, `tmin`, `aet`, `def`, `srad` (monthly) | Climate drivers — drought, runoff, evapotranspiration, temperature |
| **Ground Truth** | EY Challenge Dataset | TA, EC, DRP measurements | Prediction targets |

---

## 🔧 Feature Engineering

### Final Feature Set — 30 Features (Opt 11)

#### Landsat Spectral Features (9)

| Feature | Formula | Physical Meaning |
|---------|---------|-----------------|
| `nir` | Raw band | Near-infrared reflectance — water absorbs strongly, vegetation reflects |
| `green` | Raw band | Green reflectance — sensitive to algal chlorophyll |
| `swir16` | Raw band | Short-wave infrared 1.6µm — soil moisture, mineral content |
| `swir22` | Raw band | Short-wave infrared 2.2µm — clay minerals, suspended sediment |
| `NDMI` | `(nir - swir16) / (nir + swir16)` | Normalised Difference Moisture Index |
| `MNDWI` | `(green - swir16) / (green + swir16)` | Modified Normalised Difference Water Index |
| `NDWI` | `(green - nir) / (green + nir)` | Open water detection |
| `swir_ratio` | `swir22 / swir16` | Mineral and sediment content proxy |
| `green_nir_ratio` | `green / nir` | Vegetation density proxy |

#### TerraClimate Features (8)

| Feature | Description |
|---------|-------------|
| `pet` | Potential evapotranspiration (monthly) |
| `ppt` | Precipitation (monthly) |
| `soil` | Soil moisture (monthly) |
| `tmax` / `tmin` | Maximum / minimum temperature |
| `aet` | Actual evapotranspiration |
| `def` | Climate water deficit |
| `srad` | Solar radiation |

#### Derived Climate Features (4)

| Feature | Formula | Physical Meaning |
|---------|---------|-----------------|
| `temp_range` | `tmax - tmin` | Diurnal temperature variability |
| `water_balance` | `ppt - pet` | Net water surplus or deficit |
| `aridity_index` | `pet / ppt` | Evaporative demand relative to supply |
| `soil_saturation` | `soil / ppt` | Soil water retention relative to rainfall |

#### Temporal Features (3)

| Feature | Description |
|---------|-------------|
| `month` | Calendar month (1–12) |
| `season` | Southern hemisphere season: `(month % 12 + 3) // 3` |
| `day_of_year` | Julian day from Sample Date |

#### ⭐ Temporal Lag Features — The Breakthrough (6)

> **Added +0.019 to leaderboard score. The single biggest improvement in the project.**

Water quality does not respond instantly to climate. A rainfall event 4–8 weeks ago drives today's phosphorus runoff. Low-flow periods following dry months concentrate dissolved ions, raising conductivity. These lags encode genuinely new causal information unavailable from same-month climate variables.

| Feature | Description |
|---------|-------------|
| `ppt_lag1` | Precipitation 1 month prior |
| `ppt_lag2` | Precipitation 2 months prior |
| `soil_lag1` | Soil moisture 1 month prior |
| `soil_lag2` | Soil moisture 2 months prior |
| `aet_lag1` | Actual ET 1 month prior |
| `aet_lag2` | Actual ET 2 months prior |

**Critical implementation detail:** lags are computed *within each location* using `groupby('Location_ID').shift(n)` — never across locations, which would be physically meaningless and would leak spatial information.

```python
df_sorted[f'{var}_lag{lag}'] = (
    df_sorted.groupby('Location_ID')[var].shift(lag)
)
```

### Features Tried and Abandoned

| Feature | Reason Abandoned |
|---------|-----------------|
| `Latitude`, `Longitude` | ❌ Spatial identity leakage — model memorises training site coordinates |
| `runoff_proxy = max(ppt-aet, 0)` | ❌ Multicollinear — `ppt` and `aet` are already features. Zero new information. Caused −0.096 LB drop |
| `aet_pet_ratio = aet / pet` | ❌ Same problem — both inputs already in feature set |
| DRP `log1p` target transform | ❌ Trees are scale-invariant; `expm1` amplifies errors on high-DRP unseen locations. −0.096 LB drop |
| Multi-buffer stats (50m/150m/200m) | ❌ 83% of multi-buffer data was imputed synthetic values — compressed prediction variance |
| Harmonic encoding (`sin_month`, `cos_month`) | ❌ Added noise vs simple integer month/season |
| `log_ppt`, `log_soil`, `log_pet` | ❌ Trees don't need log-transformed inputs |
| Polynomial interaction terms | ❌ Overfit — didn't generalise spatially |
| 5-model meta-stacking | ❌ Worst score (0.038) — meta-learner memorises training site patterns |

---

## 🔑 Key Engineering Tricks

### 1. KNN Spatial Imputation for Missing Landsat Data

**The problem:** 1,085 of 9,319 training rows (11.6%) had missing Landsat NIR values. Filling with global medians silently corrupted `NDWI`, `swir_ratio`, and `green_nir_ratio` for 11.6% of all training data.

**The fix:**
```python
# K=7 spatial-temporal KNN trained on rows WITH valid NIR
# Features: [Latitude, Longitude, month] + 8 TerraClimate variables
knn_imp = KNeighborsRegressor(n_neighbors=7, weights='distance', n_jobs=-1)
knn_imp.fit(knn_X_scaled, knn_y_train)  # predict [nir, green, swir16, swir22, NDMI, MNDWI]

# CRITICAL: derive features AFTER imputation, not before
df['NDWI'] = (df['green'] - df['nir']) / (df['green'] + df['nir'] + eps)
```

**Impact:** CV score jumped from 0.22 → 0.26+ immediately after this fix was applied.

### 2. Spatial Cross-Validation (GroupKFold)

Standard KFold leaks spatial information — the same location can appear in both train and validation folds. GroupKFold ensures entire locations are held out:

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
location_ids = df['Location_ID']  # Lat_Lon string key

for tr_idx, va_idx in gkf.split(X, y, groups=location_ids):
    # No location ever appears in both train and val
    ...
```

This made CV scores predictive of leaderboard performance. The CV→LB gap stabilised at a consistent **+0.090** offset.

### 3. 5-Seed Ensemble Averaging

Training the same model with 5 different random seeds and averaging predictions reduces variance at zero added complexity:

```python
SEEDS = [42, 100, 200, 300, 400]
val_preds_all = []
for seed in SEEDS:
    et = ExtraTreesRegressor(**ET_PARAMS, random_state=seed)
    rf = RandomForestRegressor(**RF_PARAMS, random_state=seed)
    ...
    val_preds_all.append(seed_preds)
final = np.clip(np.mean(val_preds_all, axis=0), 0, None)
```

### 4. CV-Gated Feature Selection

New features are only kept if they improve spatial CV by ≥0.002 — a hard threshold to prevent adding noise that looks good in-sample but degrades leaderboard performance:

```python
THRESHOLD = 0.002
if cv_B >= cv_A + THRESHOLD:
    FEATS_FINAL = FEATS_B   # new features pass
else:
    FEATS_FINAL = FEATS_A   # revert to baseline
```

---

## 📊 Full Leaderboard History

```
Score
0.40 │
     │
0.36 │                                          ████ 0.3679 ← BEST (Opt 11)
     │                                    ████
0.32 │                              ████  0.3489 (Opt 9)
     │                        ████
0.28 │                  ████
     │            ████
0.24 │      ████
     │
0.12 │████ ← Baseline
     └──────────────────────────────────────────────────────
      Base  1   2   3   4   5   6   7   8   9  10  11
```

| Opt | Score | Strategy | Outcome |
|-----|-------|----------|---------|
| Baseline | 0.1239 | 4 features, basic RF | Starting point |
| Opt 1 | 0.263 | 13 features, RF leaf=5 | ✅ More features helped |
| Opt 2 | 0.194 | LightGBM + RF ensemble | ❌ Boosting hurt |
| Opt 3 | 0.075 | Spatial stacking with lat/lon | ❌ Coordinate leakage |
| Opt 4 | 0.038 | 5-model meta-stack | ❌ Worst score — complexity kills |
| Opt 5 | 0.271 | RF leaf=8 | ✅ Better regularisation |
| Opt 6 | 0.115 | Ridge regression | ❌ Linear underfits |
| Opt 7 | 0.279 | RF leaf=10, 100% data | ✅ Sweet-spot regularisation |
| Opt 8 *(new nb)* | — | KNN imputation pipeline | ✅ Fixed 1,085 corrupted rows |
| Opt 9 *(new nb)* | **0.3489** | 24 features, ET65+RF35, KNN fix, 3 seeds | ✅ **New best at time** |
| Opt 10 *(new nb)* | 0.253 | + runoff_proxy + DRP log1p + 5 seeds | ❌ −0.096 from multicollinear features + log transform |
| **Opt 11 *(new nb)*** | **0.3679** | Opt 9 + temporal lags (1-2mo) + 5 seeds | ✅ **BEST — +0.019 from lags** |

> **Note:** The notebook was restructured mid-project. "Old notebook" Opts 1–23 used a different base pipeline. "New notebook" Opts 8–11 build on the KNN-fixed data and Spatial GroupKFold CV.

---

## 🧠 Model Architecture

```
Input (30 features)
       │
       ▼
┌──────────────────────────────────────────┐
│           StandardScaler                 │
└──────────────────────────────────────────┘
       │
       ├──────────────────────┐
       ▼                      ▼
┌─────────────┐        ┌─────────────┐
│ ExtraTrees  │        │RandomForest │
│ 400 trees   │        │ 300 trees   │
│ depth=20    │        │ depth=18    │
│ leaf=10     │        │ leaf=10     │
│ sqrt feats  │        │ sqrt feats  │
└──────┬──────┘        └──────┬──────┘
       │   65%          35%   │
       └──────────┬───────────┘
                  ▼
          Weighted Blend
                  │
     ┌────────────┴────────────┐
     │   5-Seed Average        │
     │  seeds: 42,100,200,     │
     │         300,400         │
     └────────────┬────────────┘
                  ▼
          clip(predictions, 0)   ← negative water quality is impossible
                  │
                  ▼
          Submission (200 rows)
```

**Key hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `min_samples_leaf` | **10** | The spatial sweet spot. leaf=5 overfits, leaf=15 under-regularises |
| ET weight | **65%** | Random splits generalise better than greedy RF splits |
| RF weight | **35%** | Adds diversity to the blend |
| Seeds | **5** | Free variance reduction |

---

## 📉 The Spatial Overfitting Problem

This challenge has a fundamental difficulty: **the evaluation set contains river locations not seen during training.**

| Metric | Typical Value |
|--------|--------------|
| Internal CV R² | ~0.45 |
| Leaderboard Score | ~0.35 |
| **Gap** | **~0.10** |

Any model that memorises site-specific patterns (via coordinates, highly local thresholds, or overfitted features) will have a large gap. After implementing GroupKFold spatial CV, the gap stabilised at a predictable **+0.090 offset**:

```
Opt 11 CV: 0.2775  →  Predicted LB: 0.2775 + 0.090 = 0.3675  →  Actual LB: 0.3679  ✅
```

This predictability means CV numbers became a reliable guide for feature selection decisions.

---

## 💡 Key Lessons

### What Works
1. **Keep ALL data sources** — Landsat + TerraClimate together. Dropping Landsat collapsed scores by >0.15 (Opts 20–22 proved this)
2. **ExtraTrees > RandomForest** — random splits generalise better spatially
3. **Simple 2-model ensemble > complex stacking** — 65% ET + 35% RF beat every 3+ model stack
4. **`min_samples_leaf=10`** — the regularisation sweet spot
5. **Raw features > engineered features** — log, harmonic, and interaction transforms consistently hurt
6. **Temporal lags** — the only engineered features that genuinely helped, because they encode *new causal information* (climate memory), not just mathematical transforms of existing columns
7. **GroupKFold spatial CV** — makes CV scores predictive; without it you're flying blind

### What Fails
- **Stacking / meta-learners** — memorise training site identity
- **Boosting** (LightGBM, HistGBR) — consistently underperforms bagging on spatial tasks
- **Coordinate features** (lat/lon) — pure leakage
- **Multicollinear derived features** — if you can compute it from existing features, the tree already can too
- **Log-transforming targets** — trees are scale-invariant; `expm1` on predictions amplifies spatial errors

---

## 🏗️ Project Structure

```
├── OPTIMIZATION_8.ipynb       # KNN imputation pipeline + Opt 9/10/11 code
├── MODELING.ipynb             # Early experiments (Opt 1–23, old pipeline)
├── optimization_11.py         # Best submission — Opt 11 standalone script
├── optimization_12.py         # Next experiment — extended lags (lag-3 + pet)
├── submission.csv             # Best submission file
└── README.md                  # This file
```

---

## 🔬 How to Reproduce the Best Score

The pipeline requires the following CSVs (extracted via Snowflake + Planetary Computer):

```
water_quality_training_dataset.csv
submission_template.csv
terraclimate_features_training_final.csv
terraclimate_features_validation_final.csv
landsat_features_training.csv
landsat_features_validation.csv
```

**Step 1:** Run `OPTIMIZATION_8.ipynb` Cell 1 to build `train_base_fixed` and `val_base_fixed` (KNN imputation pipeline).

**Step 2:** Run `optimization_11.py` — it inherits `train_base_fixed`/`val_base_fixed` from memory, runs spatial CV to validate temporal lags, trains the 5-seed ensemble, and writes `submission.csv`.

```bash
# Or run as standalone (after Cell 1 has populated the fixed dataframes):
python optimization_11.py
```

**Expected output:**
```
CV Set A (Opt 9 baseline) : 0.2559
CV Set B (+ lags)         : 0.2775   ← lags pass threshold
Selected                  : B (30 features)
Predicted LB              : ~0.368
Actual LB                 : 0.3679  ✅
```

---

## 🗺️ What's Next (Opt 12+)

| Priority | Experiment | Expected Gain |
|----------|-----------|---------------|
| 🔴 High | lag-3 for ppt/soil/aet | +0.005–0.015 |
| 🔴 High | Add pet to lag variables | +0.003–0.010 |
| 🟡 Medium | Multi-scale buffers (50m, 150m tested properly) | +0.005–0.015 |
| 🟡 Medium | Sentinel-2 (has red band → true NDVI) | +0.010–0.020 |
| 🟢 Low | Bayesian hyperparameter search (Optuna) | +0.002–0.008 |

---

## 👥 Team

**Maji Safi** — EY AI & Data Challenge 2026

*"Maji Safi" means "clean water" in Swahili.*