# Function to clean a GeoDataFrame
def clean_gdf(gdf):
    # Convert datetime columns to strings
    for col in gdf.columns:
        if gdf[col].dtype == 'datetime64[ns]':
            gdf.loc[:, col] = gdf[col].astype(str)
    # Remove any columns that are not JSON serializable, excluding 'geometry'
    non_serializable_cols = []
    for col in gdf.columns:
        if col == 'geometry':
            continue  # Do not remove the 'geometry' column
        try:
            json.dumps(gdf[col].iloc[0])
        except TypeError:
            non_serializable_cols.append(col)
    if non_serializable_cols:
        gdf = gdf.drop(columns=non_serializable_cols)
    return gdf

# Load the combined GeoJSON file
gdf = gpd.read_file('omsk_region.geojson')

# Ensure the CRS is set to WGS84 (EPSG:4326)
if gdf.crs is None:
    gdf.set_crs(epsg=4326, inplace=True)
else:
    gdf = gdf.to_crs(epsg=4326)

# Separate the region boundary and districts
region_name = 'Омская область'  # Adjust this if necessary
region_gdf = gdf[gdf['name'] == region_name].copy()
districts_gdf = gdf[gdf['name'] != region_name].copy()

# Clean the GeoDataFrames
region_gdf = clean_gdf(region_gdf)
districts_gdf = clean_gdf(districts_gdf)

# Separate district polygons and points
districts_polygons = districts_gdf[districts_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
districts_points = districts_gdf[districts_gdf.geometry.type.isin(['Point', 'MultiPoint'])].copy()

# Load observations data
df = pd.read_csv('data_omsk_agro_243.csv')

# Ensure 'price' and 'area' are numeric
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['area'] = pd.to_numeric(df['area'], errors='coerce')

# Calculate price per square meter
df['price_per_sqm'] = df['price'] / df['area']

# Calculate map center
# Reproject the region GeoDataFrame to UTM Zone 43N
region_gdf_proj = region_gdf.to_crs(epsg=32643)

# Calculate the centroid in the projected CRS
region_centroid_proj = region_gdf_proj.geometry.centroid.iloc[0]

# Transform the centroid back to geographic CRS (EPSG:4326)
region_centroid_geo = gpd.GeoSeries([region_centroid_proj], crs=region_gdf_proj.crs).to_crs(epsg=4326)

# Extract latitude and longitude for map initialization
map_center = [region_centroid_geo.iloc[0].y, region_centroid_geo.iloc[0].x]

# Create the Folium map
m1 = folium.Map(location=map_center, zoom_start=6)

# Add the region boundary with thick lines
if not region_gdf.empty:
    folium.GeoJson(
        region_gdf,
        name='Omsk Region Boundary',
        style_function=lambda x: {
            'color': 'black',
            'weight': 5,
            'fillOpacity': 0
        }
    ).add_to(m1)
else:
    # Create outer boundary from districts if region geometry is missing
    region_boundary = districts_polygons.unary_union.convex_hull
    folium.GeoJson(
        region_boundary,
        name='Omsk Region Boundary',
        style_function=lambda x: {
            'color': 'black',
            'weight': 5,
            'fillOpacity': 0
        }
    ).add_to(m1)

# Add the districts layer
districts_layer = folium.FeatureGroup(name='Districts')
for idx, row in districts_polygons.iterrows():
    # Create a GeoJson object for each district polygon
    geo_json = folium.GeoJson(
        data=row['geometry'],
        style_function=lambda x: {
            'color': 'blue',
            'weight': 1,
            'fillOpacity': 0
        },
        highlight_function=lambda x: {'weight': 3, 'color': 'black'}
    )
    # Add the GeoJson object to the districts layer
    geo_json.add_to(districts_layer)
    # Bind a tooltip to the GeoJson object
    folium.Tooltip(
        row['name'],
        sticky=False,
        style=('font-size:8px;')
    ).add_to(geo_json)
# Add the districts layer to the map
districts_layer.add_to(m1)

# Add the district centers (points) as a separate layer with reduced size
if not districts_points.empty:
    district_centers_layer = folium.FeatureGroup(name='District Centers')
    # For each point, add a marker with a smaller radius
    for idx, row in districts_points.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=2,  # Reduced radius for smaller markers
            color='purple',
            fill=True,
            fill_color='purple',
            fill_opacity=1.0,
            tooltip=folium.Tooltip(
                row['name'],
                sticky=False,
                style=('font-size:8px;')
            )
        ).add_to(district_centers_layer)
    # Add the district centers layer to the map
    district_centers_layer.add_to(m1)

# Add observations layer
observations_layer = folium.FeatureGroup(name='Observations')

# Create a colormap for price_per_sqm
price_min = df['price_per_sqm'].min()
price_max = df['price_per_sqm'].max()
colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'red'], vmin=price_min, vmax=price_max)

# Add observations to the map
for idx, row in df.iterrows():
    # Determine the color based on price_per_sqm
    if pd.isnull(row['price_per_sqm']):
        color = 'gray'
    else:
        color = colormap(row['price_per_sqm'])

    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        tooltip=folium.Tooltip(
            f"ID: {row['id']}<br>"
            f"Price per sqm: {row['price_per_sqm']:.2f}<br>"
            f"Price: {row['price']}<br>"
            f"Area: {row['area']}",
            sticky=False,
            style=('font-size:10px;')
        )
    ).add_to(observations_layer)

# Add the observations layer to the map
observations_layer.add_to(m1)

# Add the colormap to the map
colormap.caption = 'Price per sqm'
colormap.add_to(m1)

# Add layer control
folium.LayerControl().add_to(m1)

# Adjust tooltip font size
css = '''<style>.leaflet-tooltip {
    font-size: 10px;
}</style>'''
m1.get_root().html.add_child(folium.Element(css))

# Display the map
m1


