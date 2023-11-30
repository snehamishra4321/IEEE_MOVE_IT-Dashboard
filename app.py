import random
import math
import pandas as pd
import folium
import numpy as np
import geopandas 
import requests
import branca
from folium.plugins import HeatMap
from shapely.wkt import loads
from folium.plugins import MarkerCluster
from folium import plugins

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

#score_data = pd.read_csv('c:/Users/Christopher/Downloads/filtered_new_point_data_v1.csv')

score_data = pd.read_csv('./data/scores.csv')
tower_df = pd.read_csv('Cellular_Towers_V1.csv')
filtered_tower_df = tower_df[(tower_df['X']>min_longitude) & (tower_df['X']<max_longitude) & (
    tower_df['Y']>min_latitude) & (tower_df['Y']<max_latitude) ]

truckdata = {
    'Longitude': [-96.465841],
    'Latitude': [30.629284]
}
truck_points = pd.DataFrame(truckdata)

# Create a map centered around the first location in your data
m = folium.Map(location=[data_points['Latitude'].iloc[0], data_points['Longitude'].iloc[0]], zoom_start=12,tiles="OpenStreetMap",control_scale =True)

   

# Define a function to get a color based on the score
def get_color(score):
    
    # For scores in between, calculate a gradient from yellow to green
    normalized_score = (score - score_data['Score'].min()) / (score_data['Score'].max() - score_data['Score'].min()) *4
    r = int(255 * (1 - normalized_score**2))  # Squaring to emphasize small differences
    g = int(255 * (normalized_score**2))
    return f'rgb({r},{g},0)'

 

# lgd_txt = '<span style="color: {col};">Crowd</span>'
color = 'red'
# # Add markers for each data point
# data = folium.FeatureGroup(lgd_txt.format( txt= color+' egg', col= "#055C9D"),show = False ).add_to(m)
# for index, row in data_points.iterrows():
#     folium.Marker([row['Latitude'], row['Longitude']], opacity=0.7,
#                   icon = BeautifyIcon(
#                 prefix = "fa", icon="user", border_color="blue", text_color="blue" #,  icon_size=(5,5)
#                   )).add_to(data)

# lgd_txt = '<span style="color: {col};">Towers</span>'
# tower = folium.FeatureGroup(lgd_txt.format( txt= color+' egg', col= "#922B21"),show = False).add_to(m)
# for ind, row in filtered_tower_df.iterrows():
#     folium.Marker(location = [row['latdec'], row['londec']],icon = BeautifyIcon(
#     prefix = "fa", icon="signal", border_color="#922B21", text_color="#922B21", text_size = 100,  icon_size=(50,50), inner_icon_style="font-size:24px;padding-top:5px;",
# )).add_to(tower)

# lgd_txt = '<span style="color: {col};">Truck</span>'
# truck = folium.FeatureGroup(lgd_txt.format( txt= color+' egg', col= "#4CBB17"),show = False).add_to(m)
# for index, row in truck_points.iterrows():
#     folium.Marker(location = [row['Latitude'], row['Longitude']],icon = BeautifyIcon(
#     icon="truck", border_color="#4CBB17", text_color="#4CBB17", icon_shape=None, icon_size=(60,60), inner_icon_style="font-size:35px;padding-top:5px;",
# )).add_to(truck)

# Create a HeatMap layer using population data
# lgd_txt = '<span style="color: {col};">Crowd Heatmap</span>'
# heat_data = [[row['Latitude'], row['Longitude']] for index, row in data_points.iterrows()]
# HeatMap(heat_data, min_opacity=0.1, blur = 30, name= lgd_txt.format( txt= color+' egg', col= "#055C9D"),show = False ).add_to(m)

# Create a HeatMap layer using tower data
# lgd_txt = '<span style="color: {col};">Signal Heatmap</span>'
# heatmap_data = [[row['latdec'], row['londec']] for ind, row in filtered_tower_df.iterrows()]
# HeatMap(heatmap_data, min_opacity=0.1, blur = 30, radius = 100,name= lgd_txt.format( txt= color+' egg', col= "#055C9D"), show = False).add_to(m)

def fit_bounds(points, m):
    sw = points[['Latitude', 'Longitude']].min().values.tolist()
    ne = points[['Latitude', 'Longitude']].max().values.tolist()
    m.fit_bounds([sw, ne])

fit_bounds(data_points, m)
# m

print("Created V1 Datasets")

## ADD Original data as well 

import geopandas as gpd


# Set filepath
fp = "./data/filtered_gdf_v2.csv"

# Read file using gpd.read_file()
gdf_read = gpd.read_file(fp, 
                    GEOM_POSSIBLE_NAMES="geometry", 
                    KEEP_GEOM_COLUMNS="NO")

print(gdf_read.info())
#gdf_read['GEOID'] = gdf_read['GEOID'].apply(lambda x:int(x))
gdf_read['GEOID_12'] = gdf_read['GEOID_12'].apply(lambda x:int(x))
gdf_read['TotPop'] = gdf_read['TotPop'].apply(lambda x:int(x))
gdf_read['D1B'] = gdf_read['D1B'].apply(lambda x:float(x))
gdf_read['NatWalkInd'] = gdf_read['NatWalkInd'].apply(lambda x:float(x))
gdf_read['STATEFP_x'] = gdf_read['STATEFP_x'].astype('int64')
gdf_read['COUNTYFP_x'] = gdf_read['COUNTYFP_x'].astype('int64')
gdf_read['TRACTCE_x'] = gdf_read['TRACTCE_x'].astype('int64')
gdf_read['BLKGRPCE_x'] = gdf_read['BLKGRPCE_x'].astype('int64')

gdf = gpd.GeoDataFrame(gdf_read[['STATEFP_x', 'COUNTYFP_x', 'TRACTCE_x', 'BLKGRPCE_x',  'GEOID', 'geometry', 'TotPop', 'D1B', 'NatWalkInd','GEOID_12']])

gdf.crs = "EPSG:4326"
gdf['power'] = 1

# Set 'power' to 0 for specified 'TRACTCE_x' values
specified_tracts = [1400, 1303, 1601, 1000, 1701, 1900]
gdf.loc[gdf['TRACTCE_x'].isin(specified_tracts), 'power'] = 0
print(gdf.info())

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

# Choose discrete colors for 0 and 1
colormap_power = linear.YlGn_09.scale(0,1)



walk_ind_dict = gdf.set_index("GEOID")["NatWalkInd"]
gross_pop_density_dict = gdf.set_index("GEOID")["D1B"]
tot_pop_dict = gdf.set_index("GEOID")["TotPop"]
power_outage = gdf.set_index("GEOID")["power"]

print(gdf)

# legend_walkind = colormap_walkind.to_step(index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
# legend_totpop = colormap_totpop.to_step(index=[0, 1, 2, 3, 4, 5])
# legend_power = colormap_power.to_step(index=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


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

# colormap_walkind.caption = "Walkability Index color scale"
# colormap_walkind.add_to(m)


folium.GeoJson(
    gdf,
    name="Total Population",
    style_function=lambda feature: {
        "fillColor": colormap_totpop(tot_pop_dict[feature["properties"]["GEOID"]]),
        "color": "black",
        "weight": 1,
        "dashArray": "5, 5",
        "fillOpacity": 0.5,
    }, show = False
).add_to(m)

#colormap_totpop.caption = "Total Population color scale"
#colormap_totpop.add_to(m)


tooltip = folium.GeoJsonTooltip(
    fields=["TRACTCE_x","NatWalkInd","power"],
    aliases=["Tract:", "Walkability Index:", "Power:"],
    localize=True,
    sticky=False,
    labels=True,
    style="""
        background-color: #F0EFEF;
        border: 2px solid black;
        border-radius: 3px;
        box-shadow: 3px;
    """,
    max_width=800,
)

folium.GeoJson(
    gdf,
    name="Power",
    style_function=lambda feature: {
        "fillColor": colormap_power(power_outage[feature['properties']['GEOID']]),
        "color": "blue",
        "weight": 1,
        "dashArray": "5, 5",
        "fillOpacity": 0.5,
    },tooltip=tooltip,
    show = False,
).add_to(m)

# colormap_power.caption = "Power color scale"
# colormap_power.add_to(m)

#colormap_walkind.add_to(m)
#colormap_totpop.add_to(m)
#colormap_power.add_to(m)



icon_create_function = """\
function(cluster) {
    return L.divIcon({
    html: '<b>' + cluster.getChildCount() + '</b>',
    className: 'marker-cluster marker-cluster-large',
    iconSize: new L.Point(20, 20)
    });
}"""


from shapely.geometry import Point
from scipy.stats import boxcox

score_data['boxcox_score'], _ = boxcox(score_data['score'] + 1)  # Adding 1 to avoid issues with zero values

score_data[['longitude', 'latitude']] = score_data['Points'].str.strip('()').str.split(', ', expand=True)

# Convert the new columns to numeric
score_data['latitude'] = pd.to_numeric(score_data['latitude'])
score_data['longitude'] = pd.to_numeric(score_data['longitude'])

lats = score_data['latitude']
lons = score_data['longitude']

# Create a list of tuples with (latitude, longitude)
locations = list(zip(lats, lons))

marker_cluster = MarkerCluster(
    locations=locations,
    #popups=popups,
    name="Score points",
    overlay=True,
    control=True,
    icon_create_function=icon_create_function,
)

for i, row in score_data.iterrows():
    tooltip = f"Score: {row['boxcox_score']}"
    folium.Marker(
        location=(row['latitude'], row['longitude']),
        tooltip=tooltip,
        icon=None,  
    ).add_to(marker_cluster)


marker_cluster.add_to(m)

score_data = score_data.sample(50)
# Extract latitude, longitude, and score from your DataFrame
locations = score_data[['latitude', 'longitude', 'boxcox_score']].values

from folium import LinearColormap
# Define a color gradient based on score using LinearColormap
color_grad = LinearColormap(['cyan', 'lime', 'yellow', 'black'], vmin=0, vmax=score_data['boxcox_score'])


# color_gradient = {
#      0.0: 'cyan',
#     0.2: 'lime',
#     0.7: 'yellow',
#     #0.6: '#645099',
#     #0.8: '#493a70',
#     1.0: 'black'
# }

color_gradient = {
    0.0:"green",
    0.2:"cyan",
    0.7:"green",
    1:"red"
}
color_grad.caption = "Score color scale"
color_grad.add_to(m)


lgd_txt = '<span>Score Heatmap</span>'
# Create a HeatMap layer with the color gradient
HeatMap(locations, name= lgd_txt.format( txt= color+' egg'),show= True).add_to(m)

marker_cluster.add_to(m)

m.fit_bounds(m.get_bounds())

folium.LayerControl().add_to(m)
folium.map.LayerControl('topleft', collapsed= False).add_to(m)

import streamlit as st
from streamlit_folium import st_folium
from folium.plugins import Fullscreen

st.title('Move-It Truck Dashboard')
#st.markdown("# Move-It Truck Dashboard")

Fullscreen(position="topleft").add_to(m)
# st_data = st_folium(m, width=725)
st_folium(m, height=725,width=725, returned_objects=[])
