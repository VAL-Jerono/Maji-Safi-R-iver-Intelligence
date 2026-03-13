# CELL 7
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Multi-dimensional arrays and datasets (e.g., NetCDF, Zarr)
import xarray as xr

# Geospatial raster data handling with CRS support
import rioxarray as rxr

# Raster operations and spatial windowing
import rasterio
from rasterio.windows import Window

# Feature preprocessing and data splitting
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Planetary Computer tools for STAC API access and authentication
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load
from pystac.extensions.eo import EOExtension as eo

from datetime import date
from tqdm import tqdm
import os 

# CELL 10
Water_Quality_df=pd.read_csv('water_quality_training_dataset.csv')
Water_Quality_df.head()

# CELL 16
landsat_train_features = pd.read_csv('landsat_features_training.csv')
landsat_train_features.head()

# CELL 18
Terraclimate_df = pd.read_csv('terraclimate_features_training.csv')
Terraclimate_df.head()

# CELL 20
# Combine two datasets vertically (along columns) using pandas concat function.
def combine_two_datasets(dataset1,dataset2,dataset3):
    '''
    Returns a  vertically concatenated dataset.
    Attributes:
    dataset1 - Dataset 1 to be combined 
    dataset2 - Dataset 2 to be combined
    '''
    
    data = pd.concat([dataset1,dataset2,dataset3], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    return data

# CELL 21
# Combining ground data and final data into a single dataset.
wq_data = combine_two_datasets(Water_Quality_df, landsat_train_features, Terraclimate_df)
wq_data.head()

# CELL 23
wq_data = wq_data.fillna(wq_data.median(numeric_only=True))
wq_data.isna().sum()

# CELL 26
# Retaining only the columns for swir22, NDMI, MNDWI, pet, Total Alkalinity, Electrical Conductance and Dissolved Reactive Phosphorus Index in the dataset.
wq_data = wq_data[['swir22','NDMI','MNDWI','pet', 'Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']]

# CELL 30
def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train_scaled, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

def evaluate_model(model, X_scaled, y_true, dataset_name="Test"):
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{dataset_name} Evaluation:")
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    return y_pred, r2, rmse

# CELL 32
def run_pipeline(X, y, param_name="Parameter"):
    print(f"\n{'='*60}")
    print(f"Training Model for {param_name}")
    print(f"{'='*60}")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    # Train
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate (in-sample)
    y_train_pred, r2_train, rmse_train = evaluate_model(model, X_train_scaled, y_train, "Train")
    
    # Evaluate (out-sample)
    y_test_pred, r2_test, rmse_test = evaluate_model(model, X_test_scaled, y_test, "Test")
    
    # Return summary
    results = {
        "Parameter": param_name,
        "R2_Train": r2_train,
        "RMSE_Train": rmse_train,
        "R2_Test": r2_test,
        "RMSE_Test": rmse_test
    }
    return model, scaler, pd.DataFrame([results])

# CELL 35
X = wq_data.drop(columns=['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus'])

y_TA = wq_data['Total Alkalinity']
y_EC = wq_data['Electrical Conductance']
y_DRP = wq_data['Dissolved Reactive Phosphorus']

model_TA, scaler_TA, results_TA = run_pipeline(X, y_TA, "Total Alkalinity")
model_EC, scaler_EC, results_EC = run_pipeline(X, y_EC, "Electrical Conductance")
model_DRP, scaler_DRP, results_DRP = run_pipeline(X, y_DRP, "Dissolved Reactive Phosphorus")


# CELL 38
results_summary = pd.concat([results_TA, results_EC, results_DRP], ignore_index=True)
results_summary


# CELL 39


# CELL 42
#Reading the coordinates for the submission
test_file = pd.read_csv('submission_template.csv')
test_file.head()

# CELL 44
landsat_val_features = pd.read_csv('landsat_features_validation.csv')
landsat_val_features.head()

# CELL 45
Terraclimate_val_df = pd.read_csv('terraclimate_features_validation.csv')
Terraclimate_val_df.head()

# CELL 46
#Consolidate all the extracted bands and features in a single dataframe
val_data = pd.DataFrame({
    'Longitude': landsat_val_features['Longitude'].values,
    'Latitude': landsat_val_features['Latitude'].values,
    'Sample Date': landsat_val_features['Sample Date'].values,
    'nir': landsat_val_features['nir'].values,
    'green': landsat_val_features['green'].values,
    'swir16': landsat_val_features['swir16'].values,
    'swir22': landsat_val_features['swir22'].values,
    'NDMI': landsat_val_features['NDMI'].values,
    'MNDWI': landsat_val_features['MNDWI'].values,
    'pet': Terraclimate_val_df['pet'].values,
})

# CELL 47
# Impute the missing values
val_data = val_data.fillna(val_data.median(numeric_only=True))

# CELL 48
# Extracting specific columns (swir22, NDMI, MNDWI, pet) from the validation dataset
submission_val_data=val_data.loc[:,['swir22','NDMI','MNDWI','pet']]
submission_val_data.head()

# CELL 49
submission_val_data.shape

# CELL 50
# --- Predicting for Total Alkalinity ---
X_sub_scaled_TA = scaler_TA.transform(submission_val_data)
pred_TA_submission = model_TA.predict(X_sub_scaled_TA)

# --- Predicting for Electrical Conductance ---
X_sub_scaled_EC = scaler_EC.transform(submission_val_data)
pred_EC_submission = model_EC.predict(X_sub_scaled_EC)

# --- Predicting for Dissolved Reactive Phosphorus ---
X_sub_scaled_DRP = scaler_DRP.transform(submission_val_data)
pred_DRP_submission = model_DRP.predict(X_sub_scaled_DRP)

# CELL 51
submission_df = pd.DataFrame({
    'Longitude': test_file['Longitude'].values,
    'Latitude': test_file['Latitude'].values,
    'Sample Date': test_file['Sample Date'].values,
    'Total Alkalinity': pred_TA_submission,
    'Electrical Conductance': pred_EC_submission,
    'Dissolved Reactive Phosphorus': pred_DRP_submission
})

# CELL 52
#Displaying the sample submission dataframe
submission_df.head()

# CELL 53
#Dumping the predictions into a csv file.
submission_df.to_csv("submission.csv",index = False)

