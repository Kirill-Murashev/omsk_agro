# syvajarvirus55a243.py

import geopandas as gpd
import folium
import json
import pandas as pd
import branca.colormap as cm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np


def clean_gdf(gdf):
    """
    Clean a GeoDataFrame to ensure it can be serialized to JSON.
    """
    # Convert datetime columns to strings
    for col in gdf.columns:
        if gdf[col].dtype == 'datetime64[ns]':
            gdf[col] = gdf[col].astype(str)
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

def load_geodata(geojson_path, region_name='Омская область'):
    """
    Load the GeoJSON file and separate the region boundary and districts.
    """
    # Step 1: Read the combined GeoJSON file
    gdf = gpd.read_file(geojson_path)
    # Ensure the CRS is set to WGS84 (EPSG:4326)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)
    # Step 2: Separate the region boundary and districts
    region_gdf = gdf[gdf['name'] == region_name].copy()
    districts_gdf = gdf[gdf['name'] != region_name].copy()
    # Clean the GeoDataFrames
    region_gdf = clean_gdf(region_gdf)
    districts_gdf = clean_gdf(districts_gdf)
    # Separate district polygons and points
    districts_polygons = districts_gdf[districts_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
    districts_points = districts_gdf[districts_gdf.geometry.type.isin(['Point', 'MultiPoint'])].copy()
    return region_gdf, districts_polygons, districts_points

def calculate_map_center(region_gdf):
    """
    Calculate the centroid for map initialization.
    """
    # Step 3: Calculate the centroid for map initialization
    # Reproject the region GeoDataFrame to UTM Zone 43N
    region_gdf_proj = region_gdf.to_crs(epsg=32643)
    # Calculate the centroid in the projected CRS
    region_centroid_proj = region_gdf_proj.geometry.centroid.iloc[0]
    # Transform the centroid back to geographic CRS (EPSG:4326)
    region_centroid_geo = gpd.GeoSeries([region_centroid_proj], crs=region_gdf_proj.crs).to_crs(epsg=4326)
    # Extract latitude and longitude for map initialization
    map_center = [region_centroid_geo.iloc[0].y, region_centroid_geo.iloc[0].x]
    return map_center

def add_region_boundary(m, region_gdf, districts_polygons):
    """
    Add the region boundary to the map with thick lines.
    """
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
        ).add_to(m)
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
        ).add_to(m)
    return m

def assign_zones(districts_polygons):
    """
    Assign climatic and agricultural zones to the districts.
    """
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
    return districts_polygons

def add_districts_layer(m, districts_polygons):
    """
    Add the districts layer (without zones) to the map for reference.
    """
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
    districts_layer.add_to(m)
    return m

def add_climatic_zones_layer(m, districts_polygons):
    """
    Add the climatic zones layer to the map.
    """
    # Step 7: Define color mappings for climate zones
    climate_zone_colors = {
        'North': '#ADD8E6',  # Light Blue
        'South': '#90EE90'   # Light Green
    }
    # Step 8: Create style function for climate zones
    def climate_style_function(feature):
        zone = feature['properties']['climate_zone']
        color = climate_zone_colors.get(zone, 'gray')
        return {
            'fillColor': color,
            'color': color,
            'weight': 1,
            'fillOpacity': 0.5
        }
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
    ).add_to(m)
    return m

def add_agricultural_zones_layer(m, districts_polygons):
    """
    Add the agricultural zones layer to the map.
    """
    # Step 7: Define color mappings for agricultural zones
    agricultural_zone_colors = {
        'North': '#00008B',                   # Dark Blue
        'North Forest-Steppe': '#228B22',     # Forest Green
        'South Forest-Steppe': '#FFFF00',     # Yellow
        'Steppe': '#FFA500'                   # Orange
    }
    # Step 8: Create style function for agricultural zones
    def agricultural_style_function(feature):
        zone = feature['properties']['agricultural_zone']
        color = agricultural_zone_colors.get(zone, 'gray')
        return {
            'fillColor': color,
            'color': color,
            'weight': 1,
            'fillOpacity': 0.5
        }
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
    ).add_to(m)
    return m

def add_district_centers_layer(m, districts_points):
    """
    Add the district centers (points) as a separate layer to the map.
    """
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
        district_centers.add_to(m)
    return m

def adjust_tooltip_font_size(m, font_size='8px'):
    """
    Adjust the font size of tooltips on the map.
    """
    # Adjust tooltip font size
    css = f'''<style>.leaflet-tooltip {{
        font-size: {font_size};
    }}</style>'''
    m.get_root().html.add_child(folium.Element(css))
    return m

def create_map_m0(geojson_path):
    """
    Create and return the map m0 with the climatic and agricultural zones.
    """
    # Load geodata
    region_gdf, districts_polygons, districts_points = load_geodata(geojson_path)
    # Calculate map center
    map_center = calculate_map_center(region_gdf)
    # Create the Folium map
    m0 = folium.Map(location=map_center, zoom_start=6)
    # Add region boundary
    m0 = add_region_boundary(m0, region_gdf, districts_polygons)
    # Assign zones
    districts_polygons = assign_zones(districts_polygons)
    # Add districts layer
    m0 = add_districts_layer(m0, districts_polygons)
    # Add climatic zones layer
    m0 = add_climatic_zones_layer(m0, districts_polygons)
    # Add agricultural zones layer
    m0 = add_agricultural_zones_layer(m0, districts_polygons)
    # Add district centers layer
    m0 = add_district_centers_layer(m0, districts_points)
    # Add layer control
    folium.LayerControl().add_to(m0)
    # Adjust tooltip font size
    m0 = adjust_tooltip_font_size(m0, font_size='8px')
    return m0


def clean_gdf(gdf):
    """
    Clean a GeoDataFrame to ensure it can be serialized to JSON.
    """
    # Convert datetime columns to strings
    for col in gdf.columns:
        if gdf[col].dtype == 'datetime64[ns]':
            gdf[col] = gdf[col].astype(str)
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

def load_geodata(geojson_path, region_name='Омская область'):
    """
    Load the GeoJSON file and separate the region boundary and districts.
    """
    gdf = gpd.read_file(geojson_path)
    # Ensure the CRS is set to WGS84 (EPSG:4326)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)
    # Separate the region boundary and districts
    region_gdf = gdf[gdf['name'] == region_name].copy()
    districts_gdf = gdf[gdf['name'] != region_name].copy()
    # Clean the GeoDataFrames
    region_gdf = clean_gdf(region_gdf)
    districts_gdf = clean_gdf(districts_gdf)
    # Separate district polygons and points
    districts_polygons = districts_gdf[districts_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
    districts_points = districts_gdf[districts_gdf.geometry.type.isin(['Point', 'MultiPoint'])].copy()
    return region_gdf, districts_polygons, districts_points

def calculate_map_center(region_gdf):
    """
    Calculate the centroid for map initialization.
    """
    # Reproject the region GeoDataFrame to UTM Zone 43N
    region_gdf_proj = region_gdf.to_crs(epsg=32643)
    # Calculate the centroid in the projected CRS
    region_centroid_proj = region_gdf_proj.geometry.centroid.iloc[0]
    # Transform the centroid back to geographic CRS (EPSG:4326)
    region_centroid_geo = gpd.GeoSeries([region_centroid_proj], crs=region_gdf_proj.crs).to_crs(epsg=4326)
    # Extract latitude and longitude for map initialization
    map_center = [region_centroid_geo.iloc[0].y, region_centroid_geo.iloc[0].x]
    return map_center

def add_region_boundary(m, region_gdf, districts_polygons):
    """
    Add the region boundary to the map with thick lines.
    """
    if not region_gdf.empty:
        folium.GeoJson(
            region_gdf,
            name='Omsk Region Boundary',
            style_function=lambda x: {
                'color': 'black',
                'weight': 5,
                'fillOpacity': 0
            }
        ).add_to(m)
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
        ).add_to(m)
    return m

def add_districts_layer(m, districts_polygons):
    """
    Add the districts layer (without zones) to the map for reference.
    """
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
    districts_layer.add_to(m)
    return m

def add_district_centers_layer(m, districts_points):
    """
    Add the district centers (points) as a separate layer to the map.
    """
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
        district_centers_layer.add_to(m)
    return m

def adjust_tooltip_font_size(m, font_size='10px'):
    """
    Adjust the font size of tooltips on the map.
    """
    css = f'''<style>.leaflet-tooltip {{
        font-size: {font_size};
    }}</style>'''
    m.get_root().html.add_child(folium.Element(css))
    return m

def load_observations(csv_path):
    """
    Load observations data and calculate price per square meter.
    """
    df = pd.read_csv(csv_path)
    # Ensure 'price' and 'area' are numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['area'] = pd.to_numeric(df['area'], errors='coerce')
    # Calculate price per square meter
    df['price_per_sqm'] = df['price'] / df['area']
    return df

def add_observations_layer(m, df):
    """
    Add observations layer to the map.
    """
    # Create a colormap for price_per_sqm
    price_min = df['price_per_sqm'].min()
    price_max = df['price_per_sqm'].max()
    colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'red'], vmin=price_min, vmax=price_max)

    # Create observations layer
    observations_layer = folium.FeatureGroup(name='Observations')

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
    observations_layer.add_to(m)
    # Add the colormap to the map
    colormap.caption = 'Price per sqm'
    colormap.add_to(m)
    return m

def create_map_m1(geojson_path, csv_path):
    """
    Create and return the map m1 with observations layer.
    """
    # Load geodata
    region_gdf, districts_polygons, districts_points = load_geodata(geojson_path)
    # Load observations data
    df = load_observations(csv_path)
    # Calculate map center
    map_center = calculate_map_center(region_gdf)
    # Create the Folium map
    m1 = folium.Map(location=map_center, zoom_start=6)
    # Add region boundary
    m1 = add_region_boundary(m1, region_gdf, districts_polygons)
    # Add districts layer
    m1 = add_districts_layer(m1, districts_polygons)
    # Add district centers layer
    m1 = add_district_centers_layer(m1, districts_points)
    # Add observations layer
    m1 = add_observations_layer(m1, df)
    # Add layer control
    folium.LayerControl().add_to(m1)
    # Adjust tooltip font size
    m1 = adjust_tooltip_font_size(m1, font_size='10px')
    return m1


def clean_gdf(gdf):
    """
    Clean a GeoDataFrame to ensure it can be serialized to JSON.
    """
    # Convert datetime columns to strings
    for col in gdf.columns:
        if gdf[col].dtype == 'datetime64[ns]':
            gdf[col] = gdf[col].astype(str)
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

def load_geodata(geojson_path, region_name='Омская область'):
    """
    Load the GeoJSON file and separate the region boundary and districts.
    """
    gdf = gpd.read_file(geojson_path)
    # Ensure the CRS is set to WGS84 (EPSG:4326)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)
    # Separate the region boundary and districts
    region_gdf = gdf[gdf['name'] == region_name].copy()
    districts_gdf = gdf[gdf['name'] != region_name].copy()
    # Clean the GeoDataFrames
    region_gdf = clean_gdf(region_gdf)
    districts_gdf = clean_gdf(districts_gdf)
    # Separate district polygons and points
    districts_polygons = districts_gdf[districts_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
    districts_points = districts_gdf[districts_gdf.geometry.type.isin(['Point', 'MultiPoint'])].copy()
    return region_gdf, districts_polygons, districts_points

def calculate_map_center(region_gdf):
    """
    Calculate the centroid for map initialization.
    """
    # Reproject the region GeoDataFrame to UTM Zone 43N
    region_gdf_proj = region_gdf.to_crs(epsg=32643)
    # Calculate the centroid in the projected CRS
    region_centroid_proj = region_gdf_proj.geometry.centroid.iloc[0]
    # Transform the centroid back to geographic CRS (EPSG:4326)
    region_centroid_geo = gpd.GeoSeries([region_centroid_proj], crs=region_gdf_proj.crs).to_crs(epsg=4326)
    # Extract latitude and longitude for map initialization
    map_center = [region_centroid_geo.iloc[0].y, region_centroid_geo.iloc[0].x]
    return map_center

def add_region_boundary(m, region_gdf, districts_polygons):
    """
    Add the region boundary to the map with thick lines.
    """
    if not region_gdf.empty:
        folium.GeoJson(
            region_gdf,
            name='Omsk Region Boundary',
            style_function=lambda x: {
                'color': 'black',
                'weight': 5,
                'fillOpacity': 0
            }
        ).add_to(m)
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
        ).add_to(m)
    return m

def add_districts_layer(m, districts_polygons):
    """
    Add the districts layer (without zones) to the map for reference.
    """
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
            row['name'].title(),  # Convert back to title case for display
            sticky=False,
            style=('font-size:8px;')
        ).add_to(geo_json)
    # Add the districts layer to the map
    districts_layer.add_to(m)
    return m

def add_district_centers_layer(m, districts_points):
    """
    Add the district centers (points) as a separate layer to the map.
    """
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
                    row['name'].title(),
                    sticky=False,
                    style=('font-size:8px;')
                )
            ).add_to(district_centers_layer)
        # Add the district centers layer to the map
        district_centers_layer.add_to(m)
    return m

def adjust_tooltip_font_size(m, font_size='10px'):
    """
    Adjust the font size of tooltips on the map.
    """
    css = f'''<style>.leaflet-tooltip {{
        font-size: {font_size};
    }}</style>'''
    m.get_root().html.add_child(folium.Element(css))
    return m

def load_observations(csv_path):
    """
    Load observations data and calculate price per square meter.
    """
    df = pd.read_csv(csv_path, encoding='utf-8')
    # Ensure 'price' and 'area' are numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['area'] = pd.to_numeric(df['area'], errors='coerce')
    # Calculate price per square meter
    df['price_per_sqm'] = df['price'] / df['area']
    return df

def prepare_district_price_data(districts_polygons, df):
    """
    Calculate mean and median price per sqm for each district and merge with districts_polygons.
    """
    # Standardize district names in observations data
    df['district'] = df['district'].str.strip().str.lower()
    # Standardize district names in districts_polygons
    districts_polygons['name'] = districts_polygons['name'].str.strip().str.lower()
    # Calculate mean and median price per sqm for each district
    district_prices = df.groupby('district')['price_per_sqm'].agg(['mean', 'median']).reset_index()
    # Merge the price data with districts_polygons
    districts_polygons = districts_polygons.merge(district_prices, left_on='name', right_on='district', how='left')
    return districts_polygons

def add_mean_price_layer(m, districts_polygons):
    """
    Add mean price per sqm layer to the map.
    """
    # Create color map for mean prices
    mean_min = districts_polygons['mean'].min()
    mean_max = districts_polygons['mean'].max()
    mean_colormap = cm.LinearColormap(['blue', 'green', 'yellow', 'red'], vmin=mean_min, vmax=mean_max)

    def mean_price_style(feature):
        mean_price = feature['properties']['mean']
        if pd.isnull(mean_price):
            color = 'gray'
        else:
            color = mean_colormap(mean_price)
        return {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }

    mean_price_layer = folium.GeoJson(
        data=districts_polygons,
        name='Mean Price per sqm',
        style_function=mean_price_style,
        highlight_function=lambda x: {'weight': 2, 'color': 'black'},
        tooltip=folium.GeoJsonTooltip(
            fields=['name', 'mean'],
            aliases=['District:', 'Mean Price per sqm:'],
            localize=True,
            style=('font-size:10px;'),
            sticky=False,
            labels=True
        )
    )
    mean_price_layer.add_to(m)
    # Add colormap legend
    mean_colormap.caption = 'Mean Price per sqm'
    mean_colormap.add_to(m)
    return m

def add_median_price_layer(m, districts_polygons):
    """
    Add median price per sqm layer to the map.
    """
    # Create color map for median prices
    median_min = districts_polygons['median'].min()
    median_max = districts_polygons['median'].max()
    median_colormap = cm.LinearColormap(['blue', 'green', 'yellow', 'red'], vmin=median_min, vmax=median_max)

    def median_price_style(feature):
        median_price = feature['properties']['median']
        if pd.isnull(median_price):
            color = 'gray'
        else:
            color = median_colormap(median_price)
        return {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }

    median_price_layer = folium.GeoJson(
        data=districts_polygons,
        name='Median Price per sqm',
        style_function=median_price_style,
        highlight_function=lambda x: {'weight': 2, 'color': 'black'},
        tooltip=folium.GeoJsonTooltip(
            fields=['name', 'median'],
            aliases=['District:', 'Median Price per sqm:'],
            localize=True,
            style=('font-size:10px;'),
            sticky=False,
            labels=True
        )
    )
    median_price_layer.add_to(m)
    # Add colormap legend
    median_colormap.caption = 'Median Price per sqm'
    median_colormap.add_to(m)
    return m

def create_map_m2(geojson_path, csv_path):
    """
    Create and return the map m2 with mean and median price per sqm layers.
    """
    # Load geodata
    region_gdf, districts_polygons, districts_points = load_geodata(geojson_path)
    # Load observations data
    df = load_observations(csv_path)
    # Prepare district price data
    districts_polygons = prepare_district_price_data(districts_polygons, df)
    # Calculate map center
    map_center = calculate_map_center(region_gdf)
    # Create the Folium map
    m2 = folium.Map(location=map_center, zoom_start=6)
    # Add region boundary
    m2 = add_region_boundary(m2, region_gdf, districts_polygons)
    # Add districts layer
    m2 = add_districts_layer(m2, districts_polygons)
    # Add district centers layer
    m2 = add_district_centers_layer(m2, districts_points)
    # Add mean price layer
    m2 = add_mean_price_layer(m2, districts_polygons)
    # Add median price layer
    m2 = add_median_price_layer(m2, districts_polygons)
    # Add layer control
    folium.LayerControl().add_to(m2)
    # Adjust tooltip font size
    m2 = adjust_tooltip_font_size(m2, font_size='10px')
    return m2


def calculate_descriptive_statistics(df, columns):
    """
    Calculate descriptive statistics for specified numerical columns in a DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - columns (list): A list of column names (strings) to analyze.

    Returns:
    - stats_df (DataFrame): A DataFrame containing the descriptive statistics.
    """
    # Initialize a dictionary to store the results
    stats_dict = {}

    for col in columns:
        if col in df.columns:
            # Ensure the column is numeric
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if not data.empty:
                mean = data.mean()
                std_dev = data.std()
                # Manually calculate Mean Absolute Deviation (MAD)
                mad = data.sub(mean).abs().mean()
                stats = {
                    'count': data.count(),
                    'mean': mean,
                    'median': data.median(),
                    'min': data.min(),
                    'max': data.max(),
                    'range': data.max() - data.min(),
                    'variance': data.var(),
                    'std_dev': std_dev,
                    'skewness': data.skew(),
                    'kurtosis': data.kurtosis(),
                    '5th_percentile': data.quantile(0.05),
                    '10th_percentile': data.quantile(0.10),
                    '25th_percentile': data.quantile(0.25),
                    '50th_percentile': data.quantile(0.50),
                    '75th_percentile': data.quantile(0.75),
                    '90th_percentile': data.quantile(0.90),
                    '95th_percentile': data.quantile(0.95),
                    'iqr': data.quantile(0.75) - data.quantile(0.25),  # Interquartile Range
                    'coefficient_of_variation': std_dev / mean if mean != 0 else None,
                    'mad': mad,  # Mean Absolute Deviation
                }
                stats_dict[col] = stats
            else:
                # Column has no numeric data
                stats_dict[col] = {}
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")

    # Convert the stats dictionary to a DataFrame
    stats_df = pd.DataFrame(stats_dict).transpose()
    return stats_df


def plot_kde_with_log_transform(df, column, log_base=np.e, shift_value=None, display_stats=True):
    """
    Plot KDE plots for the raw and log-transformed data with fitted normal curves.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - column (str): The name of the column to analyze.
    - log_base (float): The base of the logarithm (default is natural logarithm).
    - shift_value (float): A small constant to add to the data to handle zero or negative values.
                          If None, the function will determine an appropriate shift.
    - display_stats (bool): Whether to display summary statistics and normality tests (default is True).

    Returns:
    - None
    """

    # Check if the column exists in the DataFrame
    if column not in df.columns:
        print(f"Column '{column}' not found in the DataFrame.")
        return

    # Extract the data and drop missing values
    data = df[column].dropna()

    # Handle zero or negative values
    if (data <= 0).any():
        if shift_value is None:
            # Determine the minimum positive value and shift
            min_positive = data[data > 0].min()
            shift_value = min_positive / 2
            print(f"Data contains zero or negative values. Shifting data by {shift_value} to make all values positive.")
        data = data + shift_value
    else:
        shift_value = 0  # No shift needed

    # Log-transform the data
    if log_base == np.e:
        log_data = np.log(data)
        log_label = 'Natural Log'
    elif log_base == 10:
        log_data = np.log10(data)
        log_label = 'Log Base 10'
    elif log_base == 2:
        log_data = np.log2(data)
        log_label = 'Log Base 2'
    else:
        # Use log with custom base
        log_data = np.log(data) / np.log(log_base)
        log_label = f'Log Base {log_base}'

    # Prepare subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot KDE for raw data
    sns.kdeplot(data, ax=axes[0], color='blue', label='KDE')
    # Fit and plot normal distribution
    mu, std = stats.norm.fit(data)
    xmin, xmax = axes[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    axes[0].plot(x, p, 'k', linewidth=2, label='Normal Fit')
    axes[0].set_title(f'Raw Data of {column}')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # Plot KDE for log-transformed data
    sns.kdeplot(log_data, ax=axes[1], color='green', label='KDE')
    # Fit and plot normal distribution
    mu_log, std_log = stats.norm.fit(log_data)
    xmin_log, xmax_log = axes[1].get_xlim()
    x_log = np.linspace(xmin_log, xmax_log, 100)
    p_log = stats.norm.pdf(x_log, mu_log, std_log)
    axes[1].plot(x_log, p_log, 'k', linewidth=2, label='Normal Fit')
    axes[1].set_title(f'{log_label} Transformed Data of {column}')
    axes[1].set_xlabel(f'{log_label} of {column}')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    if display_stats:
        # Summary statistics
        print(f"\nSummary Statistics for {column}:")
        print(data.describe())

        print(f"\nSummary Statistics for {log_label} of {column}:")
        print(log_data.describe())

        # Normality tests
        print("\nNormality Tests:")

        # Shapiro-Wilk Test for raw data
        shapiro_stat, shapiro_p = stats.shapiro(data)
        print(f"\nShapiro-Wilk Test for raw data:")
        print(f"Statistic={shapiro_stat:.5f}, p-value={shapiro_p:.5f}")

        # Shapiro-Wilk Test for log-transformed data
        shapiro_stat_log, shapiro_p_log = stats.shapiro(log_data)
        print(f"\nShapiro-Wilk Test for log-transformed data:")
        print(f"Statistic={shapiro_stat_log:.5f}, p-value={shapiro_p_log:.5f}")

        # Interpretation
        alpha = 0.05
        if shapiro_p > alpha:
            print("\nRaw data looks normally distributed (fail to reject H0 at alpha=0.05).")
        else:
            print("\nRaw data does not look normally distributed (reject H0 at alpha=0.05).")

        if shapiro_p_log > alpha:
            print("Log-transformed data looks normally distributed (fail to reject H0 at alpha=0.05).")
        else:
            print("Log-transformed data does not look normally distributed (reject H0 at alpha=0.05).")

