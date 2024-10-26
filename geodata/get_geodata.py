import requests

def fetch_district_boundary(district_name, region_name='Омская область'):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:25];
    area["boundary"="administrative"]["admin_level"="4"]["name"="{region_name}"]->.searchArea;
    relation
      ["boundary"="administrative"]
      ["admin_level"="6"]
      ["name"="{district_name}"]
      (area.searchArea);
    out geom;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    # Convert to GeoDataFrame
    features = []
    for element in data['elements']:
        if 'geometry' in element:
            coords = [(node['lon'], node['lat']) for node in element['geometry']]
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [coords]
                },
                'properties': element['tags']
            })
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }
    gdf = gpd.GeoDataFrame.from_features(geojson)
    return gdf
