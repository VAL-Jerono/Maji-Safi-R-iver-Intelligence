# CELL 2
# Supress Warnings 
import warnings
warnings.filterwarnings('ignore')

# Import common GIS tools
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Import Planetary Computer tools
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load

# CELL 4
# Sample region in South Africa
# Contains Water Quality Sample Site #184 and #186 on Wilge River

lat_long = (-27.2923, 28.5365) # Lat-Lon centroid location
box_size_deg = 0.15 # Surrounding box in degrees

# CELL 5
# Calculate the Lat-Lon bounding box region
min_lon = lat_long[1]-box_size_deg/2
min_lat = lat_long[0]-box_size_deg/2
max_lon = lat_long[1]+box_size_deg/2
max_lat = lat_long[0]+box_size_deg/2
bounds = (min_lon, min_lat, max_lon, max_lat)

# CELL 6
# Define the time window
time_window="2015-01-01/2015-05-01"

# CELL 8
stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = stac.search(
    collections=["landsat-c2-l2"], 
    bbox=bounds, 
    datetime=time_window,
    query={"platform": {"in": ["landsat-7", "landsat-8"]}, "eo:cloud_cover": {"lt": 10}},
)
items = list(search.get_all_items())
print('This is the number of scenes that touch our region:',len(items))

# CELL 10
# Define the pixel resolution for the final product
# Define the scale according to our selected crs, so we will use degrees
resolution = 30  # meters per pixel 
scale = resolution / 111320.0 # degrees per pixel for CRS:4326 

# CELL 11
xx = stac_load(
    items,
    bands=["red", "green", "blue", "nir08", "swir16", "swir22", "qa_pixel"],
    crs="EPSG:4326", # Latitude-Longitude
    resolution=scale, # Degrees
    chunks={"x": 2048, "y": 2048},
    patch_url=pc.sign,
    bbox=bounds
)

# CELL 12
# Apply scaling and offsets for Landsat Collection-2 (reference below) to the spectral bands ONLY
# https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2
xx['red'] = (xx['red']*0.0000275)-0.2
xx['green'] = (xx['green']*0.0000275)-0.2
xx['blue'] = (xx['blue']*0.0000275)-0.2
xx['nir08'] = (xx['nir08']*0.0000275)-0.2
xx['swir16'] = (xx['swir16']*0.0000275)-0.2
xx['swir22'] = (xx['swir22']*0.0000275)-0.2

# CELL 13
# View the dimensions of our XARRAY and the variables
display(xx)

# CELL 15
plot_xx = xx[["red","green","blue"]].to_array()
plot_xx.plot.imshow(col='time', col_wrap=4, robust=True, vmin=0, vmax=0.3)
plt.show()

# CELL 16
# Select a time slice to view a simple RGB image and the cloud mask
# See the XARRAY dimensions above for the number of time slices (starts at 0)

time_slice = 1

# CELL 17
# Plot and RGB Real Color Image for a single date
fig, ax = plt.subplots(figsize=(8, 8))
xx.isel(time=time_slice)[["red", "green", "blue"]].to_array().plot.imshow(robust=True, ax=ax, vmin=0, vmax=0.3)
ax.set_title("RGB Real Color")
ax.axis('off')
plt.show()

# CELL 19
# To mask the pixels and find clouds or water, it is best to use the bit values of the 16-bit qa_pixel flag
# See the website above for a nice explanation of the process

bit_flags = {
            'fill': 1<<0,
            'dilated_cloud': 1<<1,
            'cirrus': 1<<2, 
            'cloud': 1<<3,
            'shadow': 1<<4, 
            'snow': 1<<5, 
            'clear': 1<<6,
            'water': 1<<7
}

# CELL 20
# Create a function that will mask pixels with a given type

def get_mask(mask, flags_list):
    
    # Create the result mask filled with zeros and the same shape as the mask
    final_mask = np.zeros_like(mask)
    
    # Loop through the flags  
    for flag in flags_list:
        
        # get the mask for each flag
        flag_mask = np.bitwise_and(mask, bit_flags[flag])
        
        # add it to the final flag
        final_mask = final_mask | flag_mask
    
    return final_mask > 0

# CELL 21
# Pick a single time slice to view a mask with clouds and water
sample_xx = xx.isel(time=time_slice)

# CELL 22
# Find the pixels that are no data (fill), clouds, cloud shadows, or water
my_mask = get_mask(sample_xx['qa_pixel'], ['fill', 'dilated_cloud', 'cirrus', 'cloud', 'shadow', 'water'])

# CELL 23
# Show only the mask (Yellow) with valid data in Purple
plt.figure(figsize=(7,7))
plt.imshow(my_mask)
plt.title("Cloud / Shadows / Water Mask > YELLOW")
plt.axis('off')
plt.show()

# CELL 25
# Calculate the mask for the entire xarray (all time slices)
full_mask = get_mask(xx['qa_pixel'], ['fill', 'dilated_cloud', 'cirrus', 'cloud', 'shadow', 'water'])

# CELL 26
# Create a "clean" dataset with the mask applied 
cleaned_data = xx.where(~full_mask)

# CELL 28
# Plot an NDVI image for a single date with few clouds
fig = plt.figure(figsize=(10, 7))
ndvi_image = (cleaned_data.nir08-cleaned_data.red)/(cleaned_data.nir08+cleaned_data.red)
ndvi_image.isel(time=time_slice).plot(vmin=0.0, vmax=0.8, cmap="RdYlGn")
plt.title("NDVI")
plt.axis('off')
plt.show()

# CELL 30
# Plot an NDMI image for a single date with few clouds
fig = plt.figure(figsize=(10, 7))
ndvi_image = (cleaned_data.nir08-cleaned_data.swir16)/(cleaned_data.nir08+cleaned_data.swir16)
ndvi_image.isel(time=time_slice).plot(vmin=-0.2, vmax=0.7, cmap="jet_r")
plt.title("NDMI")
plt.axis('off')
plt.show()

# CELL 32
# Plot an SWIR-22 image for a single date with few clouds
fig = plt.figure(figsize=(10, 7))
swir22_image = cleaned_data.swir22
swir22_image.isel(time=time_slice).plot(vmin=0.0, vmax=0.3, cmap="Greys")
plt.title("SWIR22")
plt.axis('off')
plt.show()

# CELL 33


