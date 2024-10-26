import geopandas as gpd
import os
import pandas as pd

# Specify the directory containing GeoJSON files
geojson_dir = './geodata'

# Initialize an empty list to store the GeoDataFrames
gdf_list = []

# Loop through each GeoJSON file in the directory
for file in os.listdir(geojson_dir):
    if file.endswith('.geojson'):
        file_path = os.path.join(geojson_dir, file)
        gdf = gpd.read_file(file_path)
        # Add a column with the district name derived from the file name
        district_name = os.path.splitext(file)[0]
        gdf['district_name'] = district_name
        gdf_list.append(gdf)

# Concatenate all GeoDataFrames into one
combined_gdf = pd.concat(gdf_list, ignore_index=True)

# Ensure that the Coordinate Reference System (CRS) is consistent
# All files are in EPSG:4326 (WGS84), set the CRS if not already set
if combined_gdf.crs is None:
    combined_gdf.set_crs(epsg=4326, inplace=True)
else:
    combined_gdf.to_crs(epsg=4326, inplace=True)

# Save the combined GeoDataFrame to a new GeoJSON file
combined_gdf.to_file('omsk_region.geojson', driver='GeoJSON')
