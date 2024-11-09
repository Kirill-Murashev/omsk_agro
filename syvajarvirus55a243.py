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
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import (
    norm, mannwhitneyu, rankdata,
    shapiro, t
)
import warnings
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)


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


def plot_pie_chart(df, column, title=None, figsize=(8, 8), autopct='%1.1f%%', colors=None, explode=None):
    """
    Plot a pie chart showing counts and percentages for each category in a specified column.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - column (str): The name of the categorical column to visualize.
    - title (str): The title of the pie chart (default is 'Distribution of {column}').
    - figsize (tuple): Figure size in inches (default is (8, 8)).
    - autopct (str): String format for displaying percentages (default is '%1.1f%%').
    - colors (list): List of colors for the pie chart slices (default is None).
    - explode (list): List of fractions to offset slices (default is None).

    Returns:
    - None
    """

    # Check if the column exists in the DataFrame
    if column not in df.columns:
        print(f"Column '{column}' not found in the DataFrame.")
        return

    # Get counts and percentages
    counts = df[column].value_counts()
    percentages = df[column].value_counts(normalize=True) * 100

    # Combine counts and percentages into labels
    labels = [f'{cat}\n{count} ({pct:.1f}%)' for cat, count, pct in zip(counts.index, counts.values, percentages.values)]

    # Plot the pie chart
    plt.figure(figsize=figsize)
    plt.pie(counts.values, labels=labels, autopct=None, startangle=90, colors=colors, explode=explode)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    if title is None:
        title = f'Distribution of {column}'
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_bar_chart(df, column, title=None, figsize=(10, 6), color='skyblue'):
    """
    Plot a horizontal bar chart showing counts for each category in a specified column.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - column (str): The name of the categorical column to visualize.
    - title (str): The title of the bar chart (default is 'Distribution of {column}').
    - figsize (tuple): Figure size in inches (default is (10, 6)).
    - color (str): Color of the bars.

    Returns:
    - None
    """

    # Check if the column exists in the DataFrame
    if column not in df.columns:
        print(f"Column '{column}' not found in the DataFrame.")
        return

    # Get counts
    counts = df[column].value_counts()

    # Plot the bar chart
    plt.figure(figsize=figsize)
    counts.sort_values().plot(kind='barh', color=color)
    if title is None:
        title = f'Distribution of {column}'
    plt.title(title)
    plt.xlabel('Count')
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()


def plot_isolated_analysis(df, price_column, variable_column, log_base=np.e, figsize=(14, 12)):
    """
    Plot four scatter plots between price per sqm and a variable, with combinations of raw and log-transformed data.
    Include Pearson, Spearman, and Kendall correlation coefficients and p-values.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - price_column (str): The name of the price per sqm column.
    - variable_column (str): The name of the variable to analyze.
    - log_base (float): The base of the logarithm for transformations (default is natural logarithm).
    - figsize (tuple): Figure size in inches (default is (14, 12)).

    Returns:
    - None
    """

    # Check if the columns exist in the DataFrame
    if price_column not in df.columns or variable_column not in df.columns:
        print(f"One or both columns '{price_column}' and '{variable_column}' not found in the DataFrame.")
        return

    # Extract the data and drop missing values
    data = df[[price_column, variable_column]].dropna()
    x = data[variable_column]
    y = data[price_column]

    # Handle zero or negative values for logarithms
    def prepare_log_data(series):
        if (series <= 0).any():
            # Shift data to positive
            min_positive = series[series > 0].min()
            shift_value = min_positive / 2
            series = series + shift_value
            print(f"Data in '{series.name}' contains zero or negative values. Shifting data by {shift_value:.5f}.")
        else:
            shift_value = 0
        return series, shift_value

    x_log, x_shift = prepare_log_data(x.copy())
    y_log, y_shift = prepare_log_data(y.copy())

    # Apply logarithm
    if log_base == np.e:
        x_log = np.log(x_log)
        y_log = np.log(y_log)
        log_label = 'Natural Log'
    elif log_base == 10:
        x_log = np.log10(x_log)
        y_log = np.log10(y_log)
        log_label = 'Log Base 10'
    elif log_base == 2:
        x_log = np.log2(x_log)
        y_log = np.log2(y_log)
        log_label = 'Log Base 2'
    else:
        x_log = np.log(x_log) / np.log(log_base)
        y_log = np.log(y_log) / np.log(log_base)
        log_label = f'Log Base {log_base}'

    # Prepare subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Plot configurations
    plot_configs = [
        (x, y, 'Raw Y vs. Raw X'),
        (x, y_log, f'{log_label} Y vs. Raw X'),
        (x_log, y, f'Raw Y vs. {log_label} X'),
        (x_log, y_log, f'{log_label} Y vs. {log_label} X')
    ]

    for ax, (x_data, y_data, title) in zip(axes.flatten(), plot_configs):
        # Scatter plot
        sns.scatterplot(x=x_data, y=y_data, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(variable_column if 'Raw X' in title else f'{log_label} of {variable_column}')
        ax.set_ylabel(price_column if 'Raw Y' in title else f'{log_label} of {price_column}')

        # Calculate correlations
        pearson_coef, pearson_p = stats.pearsonr(x_data, y_data)
        spearman_coef, spearman_p = stats.spearmanr(x_data, y_data)
        kendall_coef, kendall_p = stats.kendalltau(x_data, y_data)

        # Add correlation text
        textstr = (
            f"Pearson r: {pearson_coef:.3f} (p={pearson_p:.3f})\n"
            f"Spearman rho: {spearman_coef:.3f} (p={spearman_p:.3f})\n"
            f"Kendall tau: {kendall_coef:.3f} (p={kendall_p:.3f})"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        # Add regression line
        sns.regplot(x=x_data, y=y_data, ax=ax, scatter=False, color='red', line_kws={'linewidth': 1})

    plt.show()


def linear_regression_plot(
    df,
    y_column,
    x_column,
    use_log_y=False,
    use_log_x=False,
    log_base=np.e,
    figsize=(8, 6)
):
    """
    Perform linear regression analysis and plot the scatter plot with regression line,
    equation, and R² value on the plot.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - y_column (str): The name of the dependent variable (Y).
    - x_column (str): The name of the independent variable (X).
    - use_log_y (bool): Whether to use logarithmic transformation on Y (default is False).
    - use_log_x (bool): Whether to use logarithmic transformation on X (default is False).
    - log_base (float): The base of the logarithm for transformations (default is natural logarithm).
    - figsize (tuple): Figure size in inches (default is (8, 6)).

    Returns:
    - None
    """

    # Check if the columns exist in the DataFrame
    if y_column not in df.columns or x_column not in df.columns:
        print(f"One or both columns '{y_column}' and '{x_column}' not found in the DataFrame.")
        return

    # Extract the data and drop missing values
    data = df[[y_column, x_column]].dropna()
    X = data[x_column].copy()
    Y = data[y_column].copy()

    # Handle zero or negative values for logarithms
    def prepare_log_data(series):
        shift_value = 0
        if (series <= 0).any():
            # Shift data to positive
            min_positive = series[series > 0].min()
            shift_value = min_positive / 2
            series = series + shift_value
            print(f"Data in '{series.name}' contains zero or negative values. Shifting data by {shift_value:.5f}.")
        return series, shift_value

    # Apply log transformations if specified
    if use_log_x:
        X, x_shift = prepare_log_data(X)
        if log_base == np.e:
            X = np.log(X)
            x_label = f'ln({x_column})'
        elif log_base == 10:
            X = np.log10(X)
            x_label = f'log₁₀({x_column})'
        elif log_base == 2:
            X = np.log2(X)
            x_label = f'log₂({x_column})'
        else:
            X = np.log(X) / np.log(log_base)
            x_label = f'log_base{log_base}({x_column})'
    else:
        x_label = x_column

    if use_log_y:
        Y, y_shift = prepare_log_data(Y)
        if log_base == np.e:
            Y = np.log(Y)
            y_label = f'ln({y_column})'
        elif log_base == 10:
            Y = np.log10(Y)
            y_label = f'log₁₀({y_column})'
        elif log_base == 2:
            Y = np.log2(Y)
            y_label = f'log₂({y_column})'
        else:
            Y = np.log(Y) / np.log(log_base)
            y_label = f'log_base{log_base}({y_column})'
    else:
        y_label = y_column

    # Reshape X for sklearn
    X_reshaped = X.values.reshape(-1, 1)

    # Perform linear regression
    model = LinearRegression()
    model.fit(X_reshaped, Y)
    Y_pred = model.predict(X_reshaped)

    # Calculate R-squared
    r_squared = model.score(X_reshaped, Y)

    # Get regression coefficients
    intercept = model.intercept_
    slope = model.coef_[0]

    # Prepare the regression equation
    equation = f"y = {intercept:.3f} + {slope:.3f} * x"

    # Plotting
    plt.figure(figsize=figsize)
    sns.scatterplot(x=X, y=Y, color='blue', label='Data')
    plt.plot(X, Y_pred, color='red', label='Fit')

    # Annotate the equation and R-squared on the plot
    textstr = f"Regression equation:\n{equation}\n$R^2$ = {r_squared:.3f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.75, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Linear Regression: {y_label} vs. {x_label}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def perform_linear_regression(
    df,
    y_column,
    x_columns,
    use_log_y=False,
    use_log_x=False,
    log_base=np.e,
    add_constant=True,
    display_summary=True
):
    """
    Perform linear regression analysis and provide a full statistical inference.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - y_column (str): The name of the dependent variable (Y).
    - x_columns (str or list): The name(s) of the independent variable(s) (X).
    - use_log_y (bool): Whether to use logarithmic transformation on Y (default is False).
    - use_log_x (bool): Whether to use logarithmic transformation on X (default is False).
    - log_base (float): The base of the logarithm for transformations (default is natural logarithm).
    - add_constant (bool): Whether to add a constant term to the model (default is True).
    - display_summary (bool): Whether to display the summary of the regression results (default is True).

    Returns:
    - results (RegressionResultsWrapper): The fitted regression model results.
    """

    # Ensure x_columns is a list
    if isinstance(x_columns, str):
        x_columns = [x_columns]

    # Check if the columns exist in the DataFrame
    all_columns = [y_column] + x_columns
    for col in all_columns:
        if col not in df.columns:
            print(f"Column '{col}' not found in the DataFrame.")
            return

    # Extract the data and drop missing values
    data = df[all_columns].dropna()
    Y = data[y_column].copy()
    X = data[x_columns].copy()

    # Handle zero or negative values for logarithms
    def prepare_log_data(series):
        shift_value = 0
        if (series <= 0).any():
            # Shift data to positive
            min_positive = series[series > 0].min()
            shift_value = min_positive / 2
            series = series + shift_value
            print(f"Data in '{series.name}' contains zero or negative values. Shifting data by {shift_value:.5f}.")
        return series, shift_value

    # Apply log transformations if specified
    if use_log_y:
        Y, y_shift = prepare_log_data(Y)
        if log_base == np.e:
            Y = np.log(Y)
        elif log_base == 10:
            Y = np.log10(Y)
        elif log_base == 2:
            Y = np.log2(Y)
        else:
            Y = np.log(Y) / np.log(log_base)

    if use_log_x:
        for col in X.columns:
            X[col], x_shift = prepare_log_data(X[col])
            if log_base == np.e:
                X[col] = np.log(X[col])
            elif log_base == 10:
                X[col] = np.log10(X[col])
            elif log_base == 2:
                X[col] = np.log2(X[col])
            else:
                X[col] = np.log(X[col]) / np.log(log_base)

    # Add constant term if specified
    if add_constant:
        X = sm.add_constant(X)

    # Fit the regression model
    model = sm.OLS(Y, X)
    results = model.fit()

    if display_summary:
        print(results.summary())

    # Additional metrics
    y_pred = results.predict(X)
    residuals = Y - y_pred

    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))

    print("\nAdditional Model Quality Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.5f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.5f}")
    print(f"Mean Absolute Error (MAE): {mae:.5f}")

    return results


def plot_violin_comparison(df, y_column, x_column, figsize=(12, 6), log_base=np.e):
    """
    Plots paired violin plots for raw and log-transformed data of a continuous variable Y,
    grouped by a binary variable X.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - y_column (str): The name of the continuous variable (Y).
    - x_column (str): The name of the binary variable (X), with values 0 and 1.
    - figsize (tuple): The size of the figure (default is (12, 6)).
    - log_base (float): The base of the logarithm for transformation (default is natural logarithm).

    Returns:
    - None
    """

    # Check if columns exist
    if y_column not in df.columns:
        print(f"Column '{y_column}' not found in the DataFrame.")
        return
    if x_column not in df.columns:
        print(f"Column '{x_column}' not found in the DataFrame.")
        return

    # Ensure X is binary with values 0 and 1
    unique_values = df[x_column].dropna().unique()
    if set(unique_values) != {0, 1}:
        print(f"Column '{x_column}' must be binary with values 0 and 1.")
        return

    # Drop missing values
    data = df[[y_column, x_column]].dropna()

    # Prepare data for raw Y
    data_raw = data.copy()

    # Prepare data for log-transformed Y
    data_log = data.copy()

    # Handle zero or negative values in Y before log transformation
    if (data_log[y_column] <= 0).any():
        min_positive = data_log[y_column][data_log[y_column] > 0].min()
        shift_value = min_positive / 2
        data_log[y_column] = data_log[y_column] + shift_value
        print(f"Data in '{y_column}' contains zero or negative values. "
              f"Shifting data by {shift_value:.5f} before log transformation.")

    # Apply log transformation
    if log_base == np.e:
        data_log[y_column] = np.log(data_log[y_column])
        y_label_log = f'ln({y_column})'
    elif log_base == 10:
        data_log[y_column] = np.log10(data_log[y_column])
        y_label_log = f'log₁₀({y_column})'
    elif log_base == 2:
        data_log[y_column] = np.log2(data_log[y_column])
        y_label_log = f'log₂({y_column})'
    else:
        data_log[y_column] = np.log(data_log[y_column]) / np.log(log_base)
        y_label_log = f'log base {log_base} of {y_column}'

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # Plot violin plot for raw data
    sns.violinplot(x=x_column, y=y_column, data=data_raw, ax=axes[0], palette="Set2")
    axes[0].set_title(f'Distribution of {y_column} by {x_column} (Raw Data)')
    axes[0].set_xlabel(x_column)
    axes[0].set_ylabel(y_column)

    # Annotate with summary statistics
    medians_raw = data_raw.groupby(x_column)[y_column].median()
    for idx, median in enumerate(medians_raw):
        axes[0].text(idx, median, f'Median: {median:.2f}', horizontalalignment='center', color='black', weight='semibold')

    # Plot violin plot for log-transformed data
    sns.violinplot(x=x_column, y=y_column, data=data_log, ax=axes[1], palette="Set2")
    axes[1].set_title(f'Distribution of {y_label_log} by {x_column} (Log-Transformed Data)')
    axes[1].set_xlabel(x_column)
    axes[1].set_ylabel(y_label_log)

    # Annotate with summary statistics
    medians_log = data_log.groupby(x_column)[y_column].median()
    for idx, median in enumerate(medians_log):
        axes[1].text(idx, median, f'Median: {median:.2f}', horizontalalignment='center', color='black', weight='semibold')

    plt.tight_layout()
    plt.show()


def plot_kde_comparison(df, y_column, x_column, figsize=(14, 6), log_base=np.e):
    """
    Plots KDE plots for raw and log-transformed data of a continuous variable Y,
    including the whole population and each group defined by a binary variable X.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - y_column (str): The name of the continuous variable (Y).
    - x_column (str): The name of the binary variable (X), with values 0 and 1.
    - figsize (tuple): The size of the figure (default is (14, 6)).
    - log_base (float): The base of the logarithm for transformation (default is natural logarithm).

    Returns:
    - None
    """

    # Check if columns exist
    if y_column not in df.columns:
        print(f"Column '{y_column}' not found in the DataFrame.")
        return
    if x_column not in df.columns:
        print(f"Column '{x_column}' not found in the DataFrame.")
        return

    # Ensure X is binary with values 0 and 1
    unique_values = df[x_column].dropna().unique()
    if set(unique_values) != {0, 1}:
        print(f"Column '{x_column}' must be binary with values 0 and 1.")
        return

    # Drop missing values
    data = df[[y_column, x_column]].dropna()

    # Handle zero or negative values in Y before log transformation
    data_log = data.copy()
    if (data_log[y_column] <= 0).any():
        min_positive = data_log[y_column][data_log[y_column] > 0].min()
        shift_value = min_positive / 2
        data_log[y_column] = data_log[y_column] + shift_value
        print(f"Data in '{y_column}' contains zero or negative values. "
              f"Shifting data by {shift_value:.5f} before log transformation.")

    # Apply log transformation
    if log_base == np.e:
        data_log[y_column] = np.log(data_log[y_column])
        y_label_log = f'ln({y_column})'
    elif log_base == 10:
        data_log[y_column] = np.log10(data_log[y_column])
        y_label_log = f'log₁₀({y_column})'
    elif log_base == 2:
        data_log[y_column] = np.log2(data_log[y_column])
        y_label_log = f'log₂({y_column})'
    else:
        data_log[y_column] = np.log(data_log[y_column]) / np.log(log_base)
        y_label_log = f'log base {log_base} of {y_column}'

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot KDE for raw data
    ax = axes[0]
    sns.kdeplot(data[y_column], label='Whole Population', ax=ax, color='black', linewidth=2)
    sns.kdeplot(data[data[x_column]==0][y_column], label=f'{x_column}=0', ax=ax)
    sns.kdeplot(data[data[x_column]==1][y_column], label=f'{x_column}=1', ax=ax)
    # Plot normal curve for the whole population
    mean = data[y_column].mean()
    std = data[y_column].std()
    x_vals = np.linspace(data[y_column].min(), data[y_column].max(), 200)
    normal_curve = norm.pdf(x_vals, mean, std)
    # Scale the normal curve to match the KDE peak
    normal_curve *= max(sns.kdeplot(data[y_column], ax=ax).get_lines()[-1].get_data()[1]) / max(normal_curve)
    ax.plot(x_vals, normal_curve, color='red', linestyle='--', label='Normal Curve')
    ax.set_title(f'Distribution of {y_column} (Raw Data)')
    ax.set_xlabel(y_column)
    ax.set_ylabel('Density')
    ax.legend()

    # Plot KDE for log-transformed data
    ax = axes[1]
    sns.kdeplot(data_log[y_column], label='Whole Population', ax=ax, color='black', linewidth=2)
    sns.kdeplot(data_log[data_log[x_column]==0][y_column], label=f'{x_column}=0', ax=ax)
    sns.kdeplot(data_log[data_log[x_column]==1][y_column], label=f'{x_column}=1', ax=ax)
    # Plot normal curve for the whole population
    mean = data_log[y_column].mean()
    std = data_log[y_column].std()
    x_vals = np.linspace(data_log[y_column].min(), data_log[y_column].max(), 200)
    normal_curve = norm.pdf(x_vals, mean, std)
    # Scale the normal curve to match the KDE peak
    normal_curve *= max(sns.kdeplot(data_log[y_column], ax=ax).get_lines()[-1].get_data()[1]) / max(normal_curve)
    ax.plot(x_vals, normal_curve, color='red', linestyle='--', label='Normal Curve')
    ax.set_title(f'Distribution of {y_label_log} (Log-Transformed Data)')
    ax.set_xlabel(y_label_log)
    ax.set_ylabel('Density')
    ax.legend()

    plt.tight_layout()
    plt.show()


def perform_mann_whitney_tests(df, y_column, x_column, alternative='two-sided', log_base=np.e):
    """
    Performs the Mann-Whitney U-test on raw and log-transformed data of a continuous variable Y,
    grouped by a binary variable X.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - y_column (str): The name of the continuous variable (Y).
    - x_column (str): The name of the binary variable (X), with values 0 and 1.
    - alternative (str): Defines the alternative hypothesis ('two-sided', 'less', or 'greater').
    - log_base (float): The base of the logarithm for transformation (default is natural logarithm).

    Returns:
    - results (dict): A dictionary containing test results for raw and log-transformed data.
    """
    # Check if columns exist
    if y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found in the DataFrame.")
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in the DataFrame.")

    # Ensure X is binary with values 0 and 1
    unique_values = df[x_column].dropna().unique()
    if set(unique_values) != {0, 1}:
        raise ValueError(f"Column '{x_column}' must be binary with values 0 and 1.")

    # Drop missing values
    data = df[[y_column, x_column]].dropna()

    # Split data into groups based on X
    group0 = data[data[x_column] == 0][y_column]
    group1 = data[data[x_column] == 1][y_column]

    # Prepare results dictionary
    results = {}

    # Perform Mann-Whitney U-test on raw data
    try:
        u_stat_raw, p_value_raw = mannwhitneyu(group0, group1, alternative=alternative)
    except ValueError as e:
        warnings.warn(f"Error performing Mann-Whitney U-test on raw data: {e}")
        u_stat_raw, p_value_raw = np.nan, np.nan

    # Calculate effect size (rank-biserial correlation) for raw data
    n1, n2 = len(group0), len(group1)
    rank_sum = group1.rank().sum()
    effect_size_raw = 1 - (2 * u_stat_raw) / (n1 * n2)

    # Handle zero or negative values in Y before log transformation
    data_log = data.copy()
    shift_value = 0
    if (data_log[y_column] <= 0).any():
        min_positive = data_log[y_column][data_log[y_column] > 0].min()
        shift_value = min_positive / 2
        data_log[y_column] = data_log[y_column] + shift_value
        print(f"Data in '{y_column}' contains zero or negative values. "
              f"Shifting data by {shift_value:.5f} before log transformation.")

    # Apply log transformation
    if log_base == np.e:
        data_log[y_column] = np.log(data_log[y_column])
        y_label_log = f'ln({y_column})'
    elif log_base == 10:
        data_log[y_column] = np.log10(data_log[y_column])
        y_label_log = f'log₁₀({y_column})'
    elif log_base == 2:
        data_log[y_column] = np.log2(data_log[y_column])
        y_label_log = f'log₂({y_column})'
    else:
        data_log[y_column] = np.log(data_log[y_column]) / np.log(log_base)
        y_label_log = f'log base {log_base} of {y_column}'

    # Split log-transformed data into groups
    group0_log = data_log[data_log[x_column] == 0][y_column]
    group1_log = data_log[data_log[x_column] == 1][y_column]

    # Perform Mann-Whitney U-test on log-transformed data
    try:
        u_stat_log, p_value_log = mannwhitneyu(group0_log, group1_log, alternative=alternative)
    except ValueError as e:
        warnings.warn(f"Error performing Mann-Whitney U-test on log-transformed data: {e}")
        u_stat_log, p_value_log = np.nan, np.nan

    # Calculate effect size (rank-biserial correlation) for log-transformed data
    n1_log, n2_log = len(group0_log), len(group1_log)
    effect_size_log = 1 - (2 * u_stat_log) / (n1_log * n2_log)

    # Compile results
    results['raw'] = {
        'u_statistic': u_stat_raw,
        'p_value': p_value_raw,
        'effect_size': effect_size_raw,
        'group0_median': group0.median(),
        'group1_median': group1.median(),
        'n_group0': n1,
        'n_group1': n2,
        'shift_value': shift_value
    }

    results['log_transformed'] = {
        'u_statistic': u_stat_log,
        'p_value': p_value_log,
        'effect_size': effect_size_log,
        'group0_median': group0_log.median(),
        'group1_median': group1_log.median(),
        'n_group0': n1_log,
        'n_group1': n2_log,
        'shift_value': shift_value,
        'log_base': log_base
    }

    # Print summary of results
    print("Mann-Whitney U-test Results:")
    print("\nRaw Data:")
    print(f"Group 0 Median: {results['raw']['group0_median']:.4f}")
    print(f"Group 1 Median: {results['raw']['group1_median']:.4f}")
    print(f"U Statistic: {results['raw']['u_statistic']:.4f}")
    print(f"P-value: {results['raw']['p_value']:.4f}")
    print(f"Effect Size (Rank-Biserial Correlation): {results['raw']['effect_size']:.4f}")
    print(f"Sample Sizes: n0 = {results['raw']['n_group0']}, n1 = {results['raw']['n_group1']}")

    print("\nLog-Transformed Data:")
    print(f"Group 0 Median: {results['log_transformed']['group0_median']:.4f}")
    print(f"Group 1 Median: {results['log_transformed']['group1_median']:.4f}")
    print(f"U Statistic: {results['log_transformed']['u_statistic']:.4f}")
    print(f"P-value: {results['log_transformed']['p_value']:.4f}")
    print(f"Effect Size (Rank-Biserial Correlation): {results['log_transformed']['effect_size']:.4f}")
    print(f"Sample Sizes: n0 = {results['log_transformed']['n_group0']}, n1 = {results['log_transformed']['n_group1']}")
    print(f"Log Base Used: {log_base}")

    return results


def mann_whitney_effect_sizes(df, y_column, x_column, alternative='two-sided', conf_level=0.95, log_transform=False, log_base=np.e):
    """
    Performs the Mann-Whitney U-test and calculates effect sizes (Cliff's Delta and effect size r)
    for a continuous variable Y grouped by a binary variable X, with an option for logarithmic transformation.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - y_column (str): The name of the continuous variable (Y).
    - x_column (str): The name of the binary variable (X), with values 0 and 1.
    - alternative (str): Defines the alternative hypothesis ('two-sided', 'less', or 'greater').
    - conf_level (float): Confidence level for confidence interval (default is 0.95).
    - log_transform (bool): Whether to apply logarithmic transformation to Y (default is False).
    - log_base (float): The base of the logarithm for transformation (default is np.e).

    Returns:
    - results (dict): A dictionary containing test results and effect sizes.
    """
    # Check if columns exist
    if y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found in the DataFrame.")
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in the DataFrame.")

    # Ensure X is binary with values 0 and 1
    unique_values = df[x_column].dropna().unique()
    if set(unique_values) != {0, 1}:
        raise ValueError(f"Column '{x_column}' must be binary with values 0 and 1.")

    # Drop missing values
    data = df[[y_column, x_column]].dropna()

    # Apply logarithmic transformation if requested
    shift_value = 0  # To handle zero or negative values
    if log_transform:
        y_data = data[y_column].copy()
        # Handle zero or negative values
        if (y_data <= 0).any():
            min_positive = y_data[y_data > 0].min()
            shift_value = min_positive / 2
            y_data = y_data + shift_value
            print(f"Data in '{y_column}' contains zero or negative values. "
                  f"Shifting data by {shift_value:.5f} before log transformation.")
        # Apply logarithmic transformation
        if log_base == np.e:
            y_data = np.log(y_data)
            y_label = f'ln({y_column})'
        elif log_base == 10:
            y_data = np.log10(y_data)
            y_label = f'log₁₀({y_column})'
        elif log_base == 2:
            y_data = np.log2(y_data)
            y_label = f'log₂({y_column})'
        else:
            y_data = np.log(y_data) / np.log(log_base)
            y_label = f'log base {log_base} of {y_column}'
    else:
        y_data = data[y_column]
        y_label = y_column

    # Update data with transformed y_data
    data = data.copy()
    data[y_column] = y_data

    # Split data into groups based on X
    group0 = data[data[x_column] == 0][y_column].values
    group1 = data[data[x_column] == 1][y_column].values

    # Perform Mann-Whitney U-test
    try:
        u_statistic, p_value = mannwhitneyu(group0, group1, alternative=alternative)
    except ValueError as e:
        warnings.warn(f"Error performing Mann-Whitney U-test: {e}")
        u_statistic, p_value = np.nan, np.nan

    n0 = len(group0)
    n1 = len(group1)

    # Calculate z-statistic for effect size r
    # Continuity correction is applied when sample sizes are small
    mean_u = n0 * n1 / 2
    std_u = np.sqrt(n0 * n1 * (n0 + n1 + 1) / 12)
    if std_u == 0:
        warnings.warn("Standard deviation of U is zero, cannot compute z-statistic.")
        z = 0
    else:
        # Adjust U statistic for continuity correction
        if alternative == 'two-sided':
            correction = 0
        else:
            correction = 0.5
        z = (u_statistic - mean_u - correction) / std_u

    effect_size_r = z / np.sqrt(n0 + n1)

    # Calculate Cliff's Delta
    # Use all pairwise comparisons
    def calculate_cliffs_delta(group1, group2):
        n1 = len(group1)
        n2 = len(group2)
        total_pairs = n1 * n2
        if total_pairs == 0:
            warnings.warn("One of the groups is empty, cannot compute Cliff's Delta.")
            return np.nan, (np.nan, np.nan)

        # Efficient computation using broadcasting
        comparisons = np.subtract.outer(group1, group2)
        more = np.sum(comparisons > 0)
        less = np.sum(comparisons < 0)
        delta = (more - less) / total_pairs

        # Calculate confidence interval (assuming normal approximation)
        se_delta = np.sqrt((n0 + n1 + 1) / (3 * n0 * n1))
        alpha = 1 - conf_level
        z_crit = norm.ppf(1 - alpha / 2)
        delta_ci_lower = delta - z_crit * se_delta
        delta_ci_upper = delta + z_crit * se_delta

        return delta, (delta_ci_lower, delta_ci_upper)

    cliffs_delta, cliffs_delta_ci = calculate_cliffs_delta(group1, group0)

    # Prepare results
    results = {
        'u_statistic': u_statistic,
        'p_value': p_value,
        'z_statistic': z,
        'effect_size_r': effect_size_r,
        'cliffs_delta': cliffs_delta,
        'cliffs_delta_ci': cliffs_delta_ci,
        'group0_median': np.median(group0),
        'group1_median': np.median(group1),
        'n_group0': n0,
        'n_group1': n1,
        'log_transform': log_transform,
        'log_base': log_base,
        'shift_value': shift_value,
        'y_label': y_label,
    }

    # Print results
    print("Mann-Whitney U-test Results:")
    print(f"Group 0 (n={n0}) Median {y_label}: {results['group0_median']:.4f}")
    print(f"Group 1 (n={n1}) Median {y_label}: {results['group1_median']:.4f}")
    print(f"U Statistic: {results['u_statistic']:.4f}")
    print(f"P-value: {results['p_value']:.4f}")
    print(f"Z Statistic: {results['z_statistic']:.4f}")
    print(f"Effect Size r: {results['effect_size_r']:.4f}")
    print(f"Cliff's Delta: {results['cliffs_delta']:.4f}")
    print(f"Cliff's Delta CI ({conf_level*100:.1f}%): ({results['cliffs_delta_ci'][0]:.4f}, {results['cliffs_delta_ci'][1]:.4f})")
    if log_transform:
        print(f"Logarithmic transformation applied using base {log_base}.")
        if shift_value != 0:
            print(f"Data shifted by {shift_value:.5f} before log transformation.")
    else:
        print("No logarithmic transformation applied.")

    return results


def perform_roc_analysis(df, y_column, x_column, log_transform=False, log_base=np.e, pos_label=1):
    """
    Performs ROC analysis on a continuous variable to predict a binary outcome variable.
    Optionally applies logarithmic transformation to the continuous variable.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - y_column (str): The name of the continuous predictor variable.
    - x_column (str): The name of the binary outcome variable.
    - log_transform (bool): Whether to apply logarithmic transformation to y_column (default is False).
    - log_base (float): The base of the logarithm for transformation (default is natural logarithm).
    - pos_label (int or str): The label of the positive class in x_column (default is 1).

    Returns:
    - results (dict): A dictionary containing AUC, optimal threshold, and performance metrics.
    """
    # Check if columns exist
    if y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found in the DataFrame.")
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in the DataFrame.")

    # Ensure x_column is binary
    unique_values = df[x_column].dropna().unique()
    if len(unique_values) != 2:
        raise ValueError(f"Column '{x_column}' must be binary with two unique values.")

    # Drop missing values
    data = df[[y_column, x_column]].dropna()

    # Handle zero or negative values before log transformation
    shift_value = 0
    if log_transform:
        y_data = data[y_column].copy()
        if (y_data <= 0).any():
            min_positive = y_data[y_data > 0].min()
            shift_value = min_positive / 2
            y_data = y_data + shift_value
            print(f"Data in '{y_column}' contains zero or negative values. "
                  f"Shifting data by {shift_value:.5f} before log transformation.")
        # Apply log transformation
        if log_base == np.e:
            y_data = np.log(y_data)
            y_label = f'ln({y_column})'
        elif log_base == 10:
            y_data = np.log10(y_data)
            y_label = f'log₁₀({y_column})'
        elif log_base == 2:
            y_data = np.log2(y_data)
            y_label = f'log₂({y_column})'
        else:
            y_data = np.log(y_data) / np.log(log_base)
            y_label = f'log base {log_base} of {y_column}'
    else:
        y_data = data[y_column]
        y_label = y_column

    # Prepare data for ROC analysis
    X = y_data.values
    y = data[x_column].values

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y, X, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold using Youden's Index
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]

    # Predict classes at optimal threshold
    y_pred = (X >= optimal_threshold).astype(int)

    # Compute performance metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, pos_label=pos_label)
    recall = recall_score(y, y_pred, pos_label=pos_label)
    f1 = f1_score(y, y_pred, pos_label=pos_label)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.scatter(optimal_fpr, optimal_tpr, color='red', label='Optimal Threshold')
    plt.text(optimal_fpr, optimal_tpr - 0.05, f'Threshold = {optimal_threshold:.4f}', color='red')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve\nPredicting {x_column} using {y_label}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Prepare results
    results = {
        'AUC': roc_auc,
        'optimal_threshold': optimal_threshold,
        'optimal_tpr': optimal_tpr,
        'optimal_fpr': optimal_fpr,
        'accuracy': accuracy,
        'precision': precision,
        'recall (sensitivity)': recall,
        'specificity': specificity,
        'f1_score': f1,
        'confusion_matrix': cm,
        'shift_value': shift_value,
        'log_transform': log_transform,
        'log_base': log_base,
        'y_label': y_label
    }

    # Print results
    print("ROC Analysis Results:")
    print(f"AUC: {roc_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"True Positive Rate at Optimal Threshold (Sensitivity): {optimal_tpr:.4f}")
    print(f"False Positive Rate at Optimal Threshold: {optimal_fpr:.4f}")
    print(f"Specificity at Optimal Threshold: {specificity:.4f}")
    print(f"Accuracy at Optimal Threshold: {accuracy:.4f}")
    print(f"Precision at Optimal Threshold: {precision:.4f}")
    print(f"Recall (Sensitivity) at Optimal Threshold: {recall:.4f}")
    print(f"F1 Score at Optimal Threshold: {f1:.4f}")
    print("\nConfusion Matrix at Optimal Threshold:")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    if log_transform:
        print(f"\nLogarithmic transformation applied using base {log_base}.")
        if shift_value != 0:
            print(f"Data shifted by {shift_value:.5f} before log transformation.")
    else:
        print("\nNo logarithmic transformation applied.")

    return results


def perform_logistic_regression(df, y_column, x_column, log_transform=False, log_base=np.e,
                                threshold=None):
    """
    Performs logistic regression analysis on a continuous predictor variable to predict a binary outcome variable.
    Optionally applies logarithmic transformation to the predictor variable and allows setting a classification threshold.
    Generates textual output including the confusion matrix with labels.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - y_column (str): The name of the continuous predictor variable.
    - x_column (str): The name of the binary outcome variable.
    - log_transform (bool): Whether to apply logarithmic transformation to y_column (default is False).
    - log_base (float): The base of the logarithm for transformation (default is natural logarithm).
    - threshold (float): Classification threshold for predicted probabilities (default is 0.5 or optimal threshold).

    Returns:
    - results (dict): A dictionary containing model summary, coefficients, performance metrics, and other information.
    """
    # Check if columns exist
    if y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found in the DataFrame.")
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in the DataFrame.")

    # Drop missing values
    data = df[[y_column, x_column]].dropna()

    # Handle zero or negative values before log transformation
    shift_value = 0
    if log_transform:
        X = data[y_column].copy()
        if (X <= 0).any():
            min_positive = X[X > 0].min()
            shift_value = min_positive / 2
            X = X + shift_value
            print(f"Data in '{y_column}' contains zero or negative values. "
                  f"Shifting data by {shift_value:.5f} before log transformation.")
        # Apply log transformation
        if log_base == np.e:
            X = np.log(X)
            x_label = f'ln({y_column})'
        elif log_base == 10:
            X = np.log10(X)
            x_label = f'log₁₀({y_column})'
        elif log_base == 2:
            X = np.log2(X)
            x_label = f'log₂({y_column})'
        else:
            X = np.log(X) / np.log(log_base)
            x_label = f'log base {log_base} of {y_column}'
    else:
        X = data[y_column]
        x_label = y_column

    # Prepare the data
    y = data[x_column]

    # Encode the binary outcome variable if necessary
    if y.dtype == 'object' or y.dtype == 'bool':
        unique_classes = y.unique()
        if len(unique_classes) != 2:
            raise ValueError(f"Column '{x_column}' must have exactly two unique values.")
        y = y.map({unique_classes[0]: 0, unique_classes[1]: 1})
    elif set(y.unique()) != {0, 1}:
        # Map the smallest value to 0 and the largest to 1
        unique_values = sorted(y.unique())
        if len(unique_values) != 2:
            raise ValueError(f"Column '{x_column}' must have exactly two unique values.")
        y = y.map({unique_values[0]: 0, unique_values[1]: 1})

    # Add a constant term for the intercept
    X_const = sm.add_constant(X)

    # Fit logistic regression model
    model = sm.Logit(y, X_const)
    try:
        result = model.fit(disp=False)
    except Exception as e:
        print(f"Error fitting logistic regression model: {e}")
        return None

    # Get model predictions
    y_pred_prob = result.predict(X_const)

    # Determine the classification threshold
    if threshold is None:
        # Find the optimal threshold using Youden's Index
        fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        threshold = optimal_threshold
        threshold_info = f"Optimal threshold (Youden's Index): {threshold:.4f}"
    else:
        # Ensure the threshold is between 0 and 1
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1.")
        threshold_info = f"User-specified threshold: {threshold:.4f}"

    # Apply the classification threshold
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Calculate performance metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    # Compute confusion matrix with specified labels
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ROC Curve and AUC
    fpr, tpr, roc_thresholds = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Odds ratios and confidence intervals
    params = result.params
    conf = result.conf_int()
    conf['OR'] = params
    conf.columns = ['2.5%', '97.5%', 'OR']
    conf['OR'] = np.exp(conf['OR'])
    conf['2.5%'] = np.exp(conf['2.5%'])
    conf['97.5%'] = np.exp(conf['97.5%'])

    # Prepare results
    results = {
        'model_summary': result.summary2().as_text(),
        'coefficients': result.params,
        'odds_ratios': conf[['OR', '2.5%', '97.5%']],
        'AIC': result.aic,
        'BIC': result.bic,
        'log_likelihood': result.llf,
        'pseudo_R2': result.prsquared,
        'accuracy': accuracy,
        'precision': precision,
        'recall (sensitivity)': recall,
        'specificity': specificity,
        'f1_score': f1,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'threshold': threshold,
        'threshold_info': threshold_info,
        'shift_value': shift_value,
        'log_transform': log_transform,
        'log_base': log_base,
        'x_label': x_label
    }

    # Print results
    print("Logistic Regression Results:")
    print(f"Coefficients:\n{result.params}")
    print(f"\nOdds Ratios and 95% Confidence Intervals:\n{conf[['OR', '2.5%', '97.5%']]}")
    print(f"\nModel Fit Statistics:")
    print(f"Log-Likelihood: {result.llf:.4f}")
    print(f"AIC: {result.aic:.4f}")
    print(f"BIC: {result.bic:.4f}")
    print(f"Pseudo R-squared: {result.prsquared:.4f}")
    print(f"\nClassification Performance at Threshold {threshold:.4f}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"\n{threshold_info}")
    if log_transform:
        print(f"\nLogarithmic transformation applied using base {log_base}.")
        if shift_value != 0:
            print(f"Data shifted by {shift_value:.5f} before log transformation.")
    else:
        print("\nNo logarithmic transformation applied.")

    # Print confusion matrix with labels
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    print("\nConfusion Matrix:")
    print(cm_df)

    return results


def compare_groups_with_ratios(df, y_column, x_column, log_transform=False, log_base=np.e):
    """
    Divides the value of y_column of each observation in Group_1 by the value of y_column of each observation in Group_0.
    Computes the distribution of ratios, plots the KDE plot combined with normal curve,
    performs the Shapiro-Wilk test for normality, and calculates mean, median, and confidence interval of the mean.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - y_column (str): The name of the continuous variable.
    - x_column (str): The name of the binary variable (must have exactly two unique values).
    - log_transform (bool): Whether to apply logarithmic transformation to y_column (default is False).
    - log_base (float): The base of the logarithm for transformation (default is natural logarithm).

    Returns:
    - results (dict): A dictionary containing mean, median, confidence interval, and Shapiro-Wilk test results.
    """
    # Check if columns exist
    if y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found in the DataFrame.")
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in the DataFrame.")

    # Drop missing values
    data = df[[y_column, x_column]].dropna()

    # Ensure x_column is binary with exactly two unique values
    unique_values = data[x_column].unique()
    if len(unique_values) != 2:
        raise ValueError(f"Column '{x_column}' must have exactly two unique values.")
    else:
        # Map the values to 0 and 1 if they are not 0 and 1
        unique_values = sorted(unique_values)
        mapping = {unique_values[0]: 0, unique_values[1]: 1}
        data[x_column] = data[x_column].map(mapping)

    # Optionally apply log transformation to y_column
    y = data[y_column].copy()
    shift_value = 0
    if log_transform:
        if (y <= 0).any():
            min_positive = y[y > 0].min()
            shift_value = min_positive / 2
            y = y + shift_value
            print(f"Data in '{y_column}' contains zero or negative values. "
                  f"Shifting data by {shift_value:.5f} before log transformation.")
        # Apply log transformation
        if log_base == np.e:
            y_transformed = np.log(y)
        elif log_base == 10:
            y_transformed = np.log10(y)
        elif log_base == 2:
            y_transformed = np.log2(y)
        else:
            y_transformed = np.log(y) / np.log(log_base)
    else:
        y_transformed = y

    # Split the data into two groups
    group0 = y_transformed[data[x_column] == 0].values
    group1 = y_transformed[data[x_column] == 1].values

    # Remove zero values from group0 to prevent division by zero
    if (group0 == 0).any():
        group0 = group0[group0 != 0]
        if len(group0) == 0:
            raise ValueError("All values in group0 are zero after removing zeros, cannot compute ratios.")
        print("Zero values detected in group0 (denominator). These have been removed to prevent division by zero.")

    # Create all possible ratios
    ratios = group1[:, np.newaxis] / group0  # This creates a 2D array of ratios
    ratios = ratios.flatten()

    # Remove any infinities or NaNs that may arise
    ratios = ratios[np.isfinite(ratios)]

    # Calculate mean, median, and confidence interval of the mean
    mean_ratio = np.mean(ratios)
    median_ratio = np.median(ratios)
    std_ratio = np.std(ratios, ddof=1)
    n = len(ratios)
    alpha = 0.05  # 95% confidence interval
    t_crit = t.ppf(1 - alpha/2, df=n-1)
    margin_of_error = t_crit * std_ratio / np.sqrt(n)
    ci_lower = mean_ratio - margin_of_error
    ci_upper = mean_ratio + margin_of_error

    # Perform Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = shapiro(ratios)

    # Plot KDE plot of ratios with normal curve
    plt.figure(figsize=(10, 6))
    sns.kdeplot(ratios, label='Ratios KDE', shade=True, color='blue')
    # Overlay normal distribution
    x_values = np.linspace(min(ratios), max(ratios), 1000)
    normal_pdf = norm.pdf(x_values, loc=mean_ratio, scale=std_ratio)
    plt.plot(x_values, normal_pdf, label='Normal PDF', color='red', linestyle='--')
    plt.title('Distribution of Ratios')
    plt.xlabel('Ratio')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prepare results
    results = {
        'mean_ratio': mean_ratio,
        'median_ratio': median_ratio,
        'confidence_interval': (ci_lower, ci_upper),
        'shapiro_stat': shapiro_stat,
        'shapiro_p_value': shapiro_p,
        'n_ratios': n
    }

    # Print results
    print(f"Mean of Ratios: {mean_ratio:.4f}")
    print(f"Median of Ratios: {median_ratio:.4f}")
    print(f"95% Confidence Interval of Mean: ({ci_lower:.4f}, {ci_upper:.4f})")
    print(f"Shapiro-Wilk Test Statistic: {shapiro_stat:.4f}")
    print(f"Shapiro-Wilk Test p-value: {shapiro_p:.4e}")
    if shapiro_p < 0.05:
        print("The distribution of ratios is significantly different from normal (reject null hypothesis at alpha=0.05).")
    else:
        print("The distribution of ratios is not significantly different from normal (fail to reject null hypothesis at alpha=0.05).")

    return results


def compare_groups_with_differences(df, y_column, x_column, log_transform=True, log_base=np.e):
    """
    Computes the differences of y_column values between Group_1 and Group_0.
    For log-transformed data, computes differences which correspond to log ratios.
    Plots the KDE plot of these differences combined with normal curve,
    performs the Shapiro-Wilk test for normality, and calculates mean, median, and confidence interval of the mean.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - y_column (str): The name of the continuous variable.
    - x_column (str): The name of the binary variable (must have exactly two unique values).
    - log_transform (bool): Whether to apply logarithmic transformation to y_column (default is False).
    - log_base (float): The base of the logarithm for transformation (default is natural logarithm).

    Returns:
    - results (dict): A dictionary containing mean, median, confidence interval, and Shapiro-Wilk test results.
    """
    # Check if columns exist
    if y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found in the DataFrame.")
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in the DataFrame.")

    # Drop missing values
    data = df[[y_column, x_column]].dropna()

    # Ensure x_column is binary with exactly two unique values
    unique_values = data[x_column].unique()
    if len(unique_values) != 2:
        raise ValueError(f"Column '{x_column}' must have exactly two unique values.")
    else:
        # Map the values to 0 and 1 if they are not 0 and 1
        unique_values = sorted(unique_values)
        mapping = {unique_values[0]: 0, unique_values[1]: 1}
        data[x_column] = data[x_column].map(mapping)

    # Optionally apply log transformation to y_column
    y = data[y_column].copy()
    shift_value = 0
    if log_transform:
        if (y <= 0).any():
            min_positive = y[y > 0].min()
            shift_value = min_positive / 2
            y = y + shift_value
            print(f"Data in '{y_column}' contains zero or negative values. "
                  f"Shifting data by {shift_value:.5f} before log transformation.")
        # Apply log transformation
        if log_base == np.e:
            y_transformed = np.log(y)
        elif log_base == 10:
            y_transformed = np.log10(y)
        elif log_base == 2:
            y_transformed = np.log2(y)
        else:
            y_transformed = np.log(y) / np.log(log_base)
    else:
        y_transformed = y

    # Split the data into two groups
    group0 = y_transformed[data[x_column] == 0].values
    group1 = y_transformed[data[x_column] == 1].values

    # Create all possible differences
    differences = group1[:, np.newaxis] - group0  # This creates a 2D array of differences
    differences = differences.flatten()

    # Remove any NaNs that may arise
    differences = differences[np.isfinite(differences)]

    # Calculate mean, median, and confidence interval of the mean
    mean_diff = np.mean(differences)
    median_diff = np.median(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)
    alpha = 0.05  # 95% confidence interval
    t_crit = t.ppf(1 - alpha/2, df=n-1)
    margin_of_error = t_crit * std_diff / np.sqrt(n)
    ci_lower = mean_diff - margin_of_error
    ci_upper = mean_diff + margin_of_error

    # Perform Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = shapiro(differences)

    # Plot KDE plot of differences with normal curve
    plt.figure(figsize=(10, 6))
    sns.kdeplot(differences, label='Differences KDE', shade=True, color='blue')
    # Overlay normal distribution
    x_values = np.linspace(min(differences), max(differences), 1000)
    normal_pdf = norm.pdf(x_values, loc=mean_diff, scale=std_diff)
    plt.plot(x_values, normal_pdf, label='Normal PDF', color='red', linestyle='--')
    plt.title('Distribution of Differences')
    plt.xlabel('Difference')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prepare results
    results = {
        'mean_difference': mean_diff,
        'median_difference': median_diff,
        'confidence_interval': (ci_lower, ci_upper),
        'shapiro_stat': shapiro_stat,
        'shapiro_p_value': shapiro_p,
        'n_differences': n
    }

    # Print results
    print(f"Mean of Differences: {mean_diff:.4f}")
    print(f"Median of Differences: {median_diff:.4f}")
    print(f"95% Confidence Interval of Mean: ({ci_lower:.4f}, {ci_upper:.4f})")
    print(f"Shapiro-Wilk Test Statistic: {shapiro_stat:.4f}")
    print(f"Shapiro-Wilk Test p-value: {shapiro_p:.4e}")
    if shapiro_p < 0.05:
        print("The distribution of differences is significantly different from normal (reject null hypothesis at alpha=0.05).")
    else:
        print("The distribution of differences is not significantly different from normal (fail to reject null hypothesis at alpha=0.05).")

    return results


def create_binary_column(df, shape_column, binary_column_name, value_for_one, value_for_zero, na_values=None):
    """
    Creates a new binary column in the DataFrame based on the values in the shape column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing your data.
    - shape_column (str): The name of the column with shape values.
    - binary_column_name (str): The desired name for the new binary column.
    - value_for_one: The value in shape_column to consider as 1.
    - value_for_zero: The value in shape_column to consider as 0.
    - na_values (optional): A single value or a list of values in shape_column to consider as NA.

    Returns:
    - pd.DataFrame: The DataFrame with the new binary column added.
    """

    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    # Define mapping based on provided values
    mapping = {value_for_one: 1, value_for_zero: 0}

    # Map the shape_column to create the binary column
    df[binary_column_name] = df[shape_column].map(mapping)

    # Handle NA values
    if na_values is not None:
        if not isinstance(na_values, list):
            na_values = [na_values]
        # Set specified na_values to NaN in the binary column
        df.loc[df[shape_column].isin(na_values), binary_column_name] = pd.NA

    # Warn if there are unmapped values
    unmapped_values = df[shape_column][df[binary_column_name].isna() & ~df[shape_column].isin(na_values)].unique()
    if len(unmapped_values) > 0:
        print(f"Warning: The following values in '{shape_column}' were not mapped and set to NaN in '{binary_column_name}': {unmapped_values}")

    return df

