# Step 1: Read the combined GeoJSON file
gdf = gpd.read_file('omsk_region.geojson')

# Ensure the CRS is set to WGS84 (EPSG:4326)
if gdf.crs is None:
    gdf.set_crs(epsg=4326, inplace=True)
else:
    gdf = gdf.to_crs(epsg=4326)

# Step 2: Separate the region boundary and districts
region_name = 'Омская область'
region_gdf = gdf[gdf['name'] == region_name].copy()
districts_gdf = gdf[gdf['name'] != region_name].copy()

# Step 3: Calculate the centroid for map initialization
# Reproject the region GeoDataFrame to UTM Zone 43N
region_gdf_proj = region_gdf.to_crs(epsg=32643)
# Calculate the centroid in the projected CRS
region_centroid_proj = region_gdf_proj.geometry.centroid.iloc[0]
# Transform the centroid back to geographic CRS (EPSG:4326)
region_centroid_geo = gpd.GeoSeries([region_centroid_proj], crs=region_gdf_proj.crs).to_crs(epsg=4326)
# Extract latitude and longitude for map initialization
map_center = [region_centroid_geo.iloc[0].y, region_centroid_geo.iloc[0].x]

# Step 4: Create the Folium map
m0 = folium.Map(location=map_center, zoom_start=6)

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

# Clean the GeoDataFrames
region_gdf = clean_gdf(region_gdf)
districts_gdf = clean_gdf(districts_gdf)

# Separate district polygons and points
districts_polygons = districts_gdf[districts_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
districts_points = districts_gdf[districts_gdf.geometry.type.isin(['Point', 'MultiPoint'])].copy()

# Step 5: Add the region boundary with thick lines
if not region_gdf.empty:
    folium.GeoJson(
        region_gdf,
        name='Omsk Region Boundary',
        style_function=lambda x: {
            'color': 'black',
            'weight': 5,
            'fillOpacity': 0
        }
    ).add_to(m0)
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
    ).add_to(m0)

# Step 6: Assign climatic and agricultural zones
# Climatic zones
north_climate_districts = [
    'Усть-Ишимский район', 'Тевризский район', 'Тарский район', 'Большеуковский район',
    'Знаменский район', 'Седельниковский район', 'Крутинский район', 'Тюкалинский район',
    'Колосовский район', 'Большереченский район', 'Муромцевский район', 'Называевский район',
    'Саргатский район', 'Горьковский район', 'Нижнеомский район'
]
south_climate_districts = [
    'Исилькульский район', 'Москаленский район', 'Любинский район', 'Марьяновский район',
    'Азовский немецкий национальный район', 'Омский район', 'Кормиловский район', 'Калачинский район', 'Полтавский район',
    'Шербакульский район', 'Одесский район', 'Таврический район', 'Павлоградский район',
    'Русско-Полянский район', 'Нововаршавский район', 'Черлакский район', 'Оконешниковский район'
]
climate_zone_dict = {district: 'North' for district in north_climate_districts}
climate_zone_dict.update({district: 'South' for district in south_climate_districts})

# Agricultural zones
north_zone_districts = [
    'Усть-Ишимский район', 'Тевризский район', 'Тарский район', 'Большеуковский район',
    'Знаменский район', 'Седельниковский район'
]
north_forest_steppe_districts = [
    'Крутинский район', 'Тюкалинский район', 'Колосовский район', 'Большереченский район',
    'Муромцевский район', 'Называевский район', 'Саргатский район', 'Горьковский район',
    'Нижнеомский район'
]
south_forest_steppe_districts = [
    'Исилькульский район', 'Москаленский район', 'Любинский район', 'Марьяновский район',
    'Азовский немецкий национальный район', 'Омский район', 'Кормиловский район', 'Калачинский район'
]
steppe_zone_districts = [
    'Полтавский район', 'Шербакульский район', 'Одесский район', 'Таврический район',
    'Павлоградский район', 'Русско-Полянский район', 'Нововаршавский район',
    'Черлакский район', 'Оконешниковский район'
]
agricultural_zone_dict = {district: 'North' for district in north_zone_districts}
agricultural_zone_dict.update({district: 'North Forest-Steppe' for district in north_forest_steppe_districts})
agricultural_zone_dict.update({district: 'South Forest-Steppe' for district in south_forest_steppe_districts})
agricultural_zone_dict.update({district: 'Steppe' for district in steppe_zone_districts})

# Assign zones to districts_gdf
districts_polygons['climate_zone'] = districts_polygons['name'].map(climate_zone_dict)
districts_polygons['agricultural_zone'] = districts_polygons['name'].map(agricultural_zone_dict)

# Step 7: Define color mappings
climate_zone_colors = {
    'North': '#ADD8E6',  # Light Blue
    'South': '#90EE90'   # Light Green
}
agricultural_zone_colors = {
    'North': '#00008B',                   # Dark Blue
    'North Forest-Steppe': '#228B22',     # Forest Green
    'South Forest-Steppe': '#FFFF00',     # Yellow
    'Steppe': '#FFA500'                   # Orange
}

# Step 8: Create style functions
def climate_style_function(feature):
    zone = feature['properties']['climate_zone']
    color = climate_zone_colors.get(zone, 'gray')
    return {
        'fillColor': color,
        'color': color,
        'weight': 1,
        'fillOpacity': 0.5
    }

def agricultural_style_function(feature):
    zone = feature['properties']['agricultural_zone']
    color = agricultural_zone_colors.get(zone, 'gray')
    return {
        'fillColor': color,
        'color': color,
        'weight': 1,
        'fillOpacity': 0.5
    }

# Step 9: Add the districts layer (without zones) for reference
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
districts_layer.add_to(m0)

# Step 10: Add climatic zones layer
folium.GeoJson(
    data=districts_polygons,
    name='Climatic Zones',
    style_function=climate_style_function,
    highlight_function=lambda x: {'weight': 2, 'color': 'black'},
    tooltip=folium.GeoJsonTooltip(
        fields=['name', 'climate_zone'],
        aliases=['District:', 'Climatic Zone:'],
        sticky=False,
        style=('font-size:8px;')
    )
).add_to(m0)

# Step 11: Add agricultural zones layer
folium.GeoJson(
    data=districts_polygons,
    name='Agricultural Zones',
    style_function=agricultural_style_function,
    highlight_function=lambda x: {'weight': 2, 'color': 'black'},
    tooltip=folium.GeoJsonTooltip(
        fields=['name', 'agricultural_zone'],
        aliases=['District:', 'Agricultural Zone:'],
        sticky=False,
        style=('font-size:8px;')
    )
).add_to(m0)

# Step 12: Add the district centers (points) as a separate layer
if not districts_points.empty:
    district_centers = folium.FeatureGroup(name='District Centers')

    # For each point, add a marker with a smaller radius
    for idx, row in districts_points.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,  # Adjust the radius to make markers smaller
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=1.0,
            tooltip=row['name']
        ).add_to(district_centers)

    # Add the district centers layer to the map
    district_centers.add_to(m0)

# Step 13: Update layer control
folium.LayerControl().add_to(m0)

# Adjust tooltip font size
css = '''
<style>
.leaflet-tooltip {
    font-size: 8px;
}
</style>
'''
m0.get_root().html.add_child(folium.Element(css))

# Step 14: Display the map
m0
