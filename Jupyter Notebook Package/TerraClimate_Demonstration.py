# CELL 2
# Supress Warnings 
import warnings
warnings.filterwarnings('ignore')

# Import common GIS tools
import xarray as xr

# Import Planetary Computer tools
import pystac_client
import planetary_computer 

# CELL 4
# Access STAC catalog and collection.
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace)

collection = catalog.get_collection("terraclimate")
asset = collection.assets["zarr-abfs"]

# CELL 5
# Open dataset and remove CRS.
ds = xr.open_dataset(asset.href,**asset.extra_fields["xarray:open_kwargs"])
ds = ds.drop('crs', dim=None) # Remove the CRS coordinate in the dataset
ds

# CELL 7
# Since this is a HUGE dataset (nearly 2 TB), we should parse the dataset
# Trimming dataset to years 2011 through 2015
ds = ds.sel(time=slice("2011-01-01", "2015-12-31"))

# CELL 8
# Sample region in South Africa
# Contains Water Quality Sample Site #184 and #186 on Wilge River
lat_long = (-27.2923, 28.5365) # Lat-Lon centroid location
box_size_deg = 0.15 # Surrounding box in degrees

# CELL 9
# Calculate the Lat-Lon bounding box region
min_lon = lat_long[1]-box_size_deg/2
min_lat = lat_long[0]-box_size_deg/2
max_lon = lat_long[1]+box_size_deg/2
max_lat = lat_long[0]+box_size_deg/2

# CELL 10
mask_lon = (ds.lon >= min_lon) & (ds.lon <= max_lon)
mask_lat = (ds.lat >= min_lat) & (ds.lat <= max_lat)

# CELL 11
# Crop the dataset to smaller Lat-Lon regions
ds = ds.where(mask_lon & mask_lat, drop=True)
ds

# CELL 13
# Plot monthly accumulated precipitation over the region for 5 years
temperature = ds["ppt"].mean(dim=["lat", "lon"])
temperature.plot(figsize=(12, 6));

# CELL 14
# Plot monthly reference evapotransporation over the region for 5 years
temperature = ds["pet"].mean(dim=["lat", "lon"])
temperature.plot(figsize=(12, 6));

# CELL 15


