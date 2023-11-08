import random
import math
import pandas as pd
import folium
import numpy as np
import geopandas 
import requests
import branca

from folium.plugins import BeautifyIcon
from folium.plugins import MousePosition
from folium.plugins import HeatMap

# College Station
# Top Right - 30.757470, -96.249537
# Bottom Left - 30.586657, -96.461866


# Example usage:
min_latitude, max_latitude = 30.586657, 30.757470
# min_latitude, max_latitude = 30.586657, 31.757470
min_longitude, max_longitude = -96.461866, -96.249537


max_dist = 1
crowd_fraction = 0.7

num_points = 50

# Tamu a
tamu_location = (30.612178, -96.341015)
# Ross HEB
heb = (30.616305, -96.314587)
# Northgate
northgate = (30.623859, -96.347556)
# George Bush Library 
lib = (30.596121, -96.354207)
# Bryan
bryan = (30.673732, -96.369490)
# test_location = (30.612178, -96.50)

crowd_centers = [tamu_location, heb, lib, bryan]  # Example crowd centers
# crowd_centers = [tamu_location]


def generate_random_points(min_lat, max_lat, min_lon, max_lon, crowd_centers, n):
    data_points = []

    for _ in range(n):

        if random.random() < crowd_fraction:
            # Select a random crowd center
            center = random.choice(crowd_centers)

            # Generate a random distance and angle
            distance = random.uniform(0, max_dist)  # Adjust the maximum distance as needed
            angle = random.uniform(0, 2 * math.pi)

            # Convert distance and angle to latitude and longitude differentials
            lat_diff = distance * math.cos(angle) / 111.32  # 1 degree of latitude is approximately 111.32 km
            lon_diff = distance * math.sin(angle) / (111.32 * math.cos(center[0]))

            # Calculate the new latitude and longitude
            new_lat = center[0] + lat_diff
            new_lon = center[1] + lon_diff

            # Ensure the generated point is within the specified boundaries            
            new_lat = max(min_lat, min(new_lat, max_lat))
            new_lon = max(min_lon, min(new_lon, max_lon))

        else:
            new_lat = random.uniform(min_lat, max_lat)
            new_lon = random.uniform(min_lon, max_lon)

        data_points.append((new_lat, new_lon))

    return data_points

data_points = generate_random_points(min_latitude, max_latitude, min_longitude, max_longitude, crowd_centers, num_points)

# Create a DataFrame from the generated random points
data_points = pd.DataFrame(data_points, columns=['Latitude', 'Longitude'])


tower_df = pd.read_csv('./data/Cellular_Towers_V1.csv')
filtered_tower_df = tower_df[(tower_df['X']>min_longitude) & (tower_df['X']<max_longitude) & (
    tower_df['Y']>min_latitude) & (tower_df['Y']<max_latitude) ]

truckdata = {
    'Longitude': [-96.465841],
    'Latitude': [30.629284]
}
truck_points = pd.DataFrame(truckdata)

# Create a map centered around the first location in your data
m = folium.Map(location=[data_points['Latitude'].iloc[0], data_points['Longitude'].iloc[0]], zoom_start=12,tiles="OpenStreetMap",control_scale =True)

lgd_txt = '<span style="color: {col};">Crowd</span>'
color = 'red'
# Add markers for each data point
data = folium.FeatureGroup(lgd_txt.format( txt= color+' egg', col= "#055C9D") ).add_to(m)
for index, row in data_points.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], opacity=0.7,
                  icon = BeautifyIcon(
                prefix = "fa", icon="user", border_color="blue", text_color="blue" #,  icon_size=(5,5)
                  )).add_to(data)

lgd_txt = '<span style="color: {col};">Towers</span>'
tower = folium.FeatureGroup(lgd_txt.format( txt= color+' egg', col= "#922B21")).add_to(m)
for ind, row in filtered_tower_df.iterrows():
    folium.Marker(location = [row['latdec'], row['londec']],icon = BeautifyIcon(
    prefix = "fa", icon="signal", border_color="#922B21", text_color="#922B21", text_size = 100,  icon_size=(50,50), inner_icon_style="font-size:24px;padding-top:5px;",
)).add_to(tower)
    
lgd_txt = '<span style="color: {col};">Truck</span>'
truck = folium.FeatureGroup(lgd_txt.format( txt= color+' egg', col= "#4CBB17")).add_to(m)
for index, row in truck_points.iterrows():
    folium.Marker(location = [row['Latitude'], row['Longitude']],icon = BeautifyIcon(
    icon="truck", border_color="#4CBB17", text_color="#4CBB17", icon_shape=None, icon_size=(60,60), inner_icon_style="font-size:35px;padding-top:5px;",
)).add_to(truck)

# Create a HeatMap layer using population data
lgd_txt = '<span style="color: {col};">Crowd Heatmap</span>'
heat_data = [[row['Latitude'], row['Longitude']] for index, row in data_points.iterrows()]
HeatMap(heat_data, min_opacity=0.1, blur = 30, name= lgd_txt.format( txt= color+' egg', col= "#055C9D") ).add_to(m)



def fit_bounds(points, m):
    sw = points[['Latitude', 'Longitude']].min().values.tolist()
    ne = points[['Latitude', 'Longitude']].max().values.tolist()
    m.fit_bounds([sw, ne])

fit_bounds(data_points, m)
# m

print("Created V1 Datasets")

## ADD Original data as well 

import geopandas as gpd

# gdf = gpd.GeoDataFrame(pd.read_csv('data/Census_geo_data.csv'))
walkability_df = pd.read_csv('data/EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv')
walkability_df['GEOID10'] = walkability_df['GEOID10'].astype('Int64')
walkability_df['GEOID20'] = walkability_df['GEOID20'].astype('Int64')
walkability_df['STATEFP'] = walkability_df['STATEFP'].astype('Int64')
walkability_df['COUNTYFP'] = walkability_df['COUNTYFP'].astype('Int64')
walkability_df['TRACTCE'] = walkability_df['TRACTCE'].astype('Int64')
walkability_df['BLKGRPCE'] = walkability_df['BLKGRPCE'].astype('Int64')

# Create a 12 digit GEO ID to merge with Shape dataset
walkability_df['GEOID_12'] = walkability_df.apply(lambda x: int(str(x['STATEFP']) + str(x['COUNTYFP']).zfill(3) + str(x['TRACTCE']).zfill(6) + str(x['BLKGRPCE'])) , axis=1) 
texas_walkability_df = walkability_df[walkability_df['STATEFP']==48]
del walkability_df

print("Created Dataset for Walkability Index")

fp = "data/cb_2018_48_bg_500k/cb_2018_48_bg_500k.shp"

# Read file using gpd.read_file()
spatial_df = gpd.read_file(fp)
spatial_df['GEOID'] = spatial_df['GEOID'].apply(lambda x:int(x))
spatial_df['TRACTCE'] = spatial_df['TRACTCE'].apply(lambda x:int(x))
spatial_df['STATEFP'] = spatial_df['STATEFP'].apply(lambda x:int(x))
spatial_df['COUNTYFP'] = spatial_df['COUNTYFP'].apply(lambda x:int(x))
spatial_df['BLKGRPCE'] = spatial_df['BLKGRPCE'].apply(lambda x:int(x))
df  = spatial_df.merge(texas_walkability_df, left_on='GEOID', right_on='GEOID_12', how='left')
# df  = tab_df.merge(spatial_df, on='mukey', how='right')
gdf = gpd.GeoDataFrame(df[['STATEFP_x', 'COUNTYFP_x', 'TRACTCE_x', 'BLKGRPCE_x',  'GEOID', 'geometry', 'TotPop', 'D1B', 'NatWalkInd','GEOID_12']])
del df
print("Created Dataset for Geolocation")
from branca.colormap import linear

# Use dir(linear) to find all possible colors

colormap_walkind = linear.YlGn_09.scale(
    gdf.NatWalkInd.min(), gdf.NatWalkInd.max()
)

colormap_gross_pop_density = branca.colormap.LinearColormap(colors=['white', 'yellow', 'orange', 'red'],
                                          index = np.round(np.linspace(gdf.D1B.min(), gdf.D1B.max()/5, 4)),
                                          vmin = gdf.D1B.min(), vmax = gdf.D1B.max(), tick_labels = np.round(np.exp(np.linspace(gdf.D1B.min(), gdf.D1B.max(), 4)),1)
           )

colormap_totpop = branca.colormap.LinearColormap(colors=['white', 'yellow', 'orange', 'red'],
                                          index = np.round(np.linspace(gdf.TotPop.min(), gdf.TotPop.max()/5, 4)),
                                          vmin = gdf.TotPop.min(), vmax = gdf.TotPop.max(), tick_labels = np.round(np.exp(np.linspace(gdf.TotPop.min(), gdf.TotPop.max(), 4)),1)
           )

walk_ind_dict = gdf.set_index("GEOID")["NatWalkInd"]
gross_pop_density_dict = gdf.set_index("GEOID")["D1B"]
tot_pop_dict = gdf.set_index("GEOID")["TotPop"]
folium.GeoJson(
    gdf,
    name="Walkability Index",
    style_function=lambda feature: {
        "fillColor": colormap_walkind(walk_ind_dict[feature["properties"]["GEOID"]]),
        "color": "black",
        "weight": 1,
        "dashArray": "5, 5",
        "fillOpacity": 0.5,
    },
).add_to(m)

# folium.GeoJson(
#     gdf,
#     name="Gross Population Density",
#     style_function=lambda feature: {
#         "fillColor": colormap_gross_pop_density(gross_pop_density_dict[feature["properties"]["GEOID"]]),
#         "color": "black",
#         "weight": 1,
#         "dashArray": "5, 5",
#         "fillOpacity": 0.5,
#     },
# ).add_to(m)

folium.GeoJson(
    gdf,
    name="Total Population",
    style_function=lambda feature: {
        "fillColor": colormap_totpop(tot_pop_dict[feature["properties"]["GEOID"]]),
        "color": "black",
        "weight": 1,
        "dashArray": "5, 5",
        "fillOpacity": 0.5,
    },
).add_to(m)


m.fit_bounds(m.get_bounds())
folium.LayerControl().add_to(m)
folium.map.LayerControl('topleft', collapsed= False).add_to(m)

from streamlit_folium import st_folium
from folium.plugins import Fullscreen

Fullscreen(position="topleft").add_to(m)
# st_data = st_folium(m, width=725)
st_folium(m, height=725,width=725, returned_objects=[])