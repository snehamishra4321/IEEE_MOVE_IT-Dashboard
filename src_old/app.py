import random
import math
import pandas as pd
import folium
import numpy as np
import geopandas 
import requests
import branca
from PIL import Image
import base64

import streamlit as st

from folium.plugins import HeatMap
from shapely.wkt import loads
from folium.plugins import MarkerCluster
from folium import plugins

from folium.plugins import BeautifyIcon
from folium.plugins import MousePosition
from folium.plugins import HeatMap
from streamlit.components.v1 import html
from folium import LinearColormap

import streamlit as st
from streamlit_folium import st_folium
from folium.plugins import Fullscreen
import geopandas as gpd
from branca.colormap import linear
from shapely.geometry import Point
from scipy.stats import boxcox
import pandas as pd 
import geopandas as gpd
import numpy as np
import pickle
from shapely import wkt

import folium
from folium.plugins import MarkerCluster
from folium.plugins import BeautifyIcon

import time
import warnings
warnings.filterwarnings("ignore")


last_logged_time = time.time()

def get_delay():
    global last_logged_time
    delay = time.time() - last_logged_time
    last_logged_time = time.time()
    return delay

def convert_list_to_gdf(list_of_tuples):
        df = gpd.GeoDataFrame([['POINT ' +str(i).replace(',', ' ')]  for i in list_of_tuples], columns=['geometry'])
        df['geometry'] = df.apply(lambda x: wkt.loads(x['geometry']), axis=1)
        df = df.set_geometry("geometry")
        df.crs = "EPSG:4326"
        return df

@st.cache_data
def get_feature_data():
    # tower_data_file = 'data/filtered_tower_df_9_counties.csv'
    tower_data_file = 'data/processed_data/tower_df.csv'
    amenities_data_file = 'data/amenities_9_counties.pkl'
    final_points_data_file = 'data/final_points.pkl'
    
    tower_df = pd.read_csv(tower_data_file)

    # Collect other Amenities Data
    with open(amenities_data_file, 'rb') as handle:
        amenities_dict = pickle.load(handle)

    

    restaurants = convert_list_to_gdf(amenities_dict['Food_locations'])
    hospitals = convert_list_to_gdf(amenities_dict['Hospital_locations'])
    fire_stations = convert_list_to_gdf(amenities_dict['Fire_station_locations'])
    parkings = convert_list_to_gdf(amenities_dict['Car_parking_locations'])
    

    return restaurants, hospitals, fire_stations, parkings, tower_df #tower_coordinates


@st.cache_data
def get_gdf_data():
    # Set filepath
    fp = "./data/filtered_gdf_v2.csv"
    # Read file using gpd.read_file()
    gdf_read = gpd.read_file(fp, 
                        GEOM_POSSIBLE_NAMES="geometry", 
                        KEEP_GEOM_COLUMNS="NO")

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
    gdf['power'] = 0
    # Set 'power' to 0 for specified 'TRACTCE_x' values
    specified_tracts = [1400, 1303, 1601, 1000, 1701, 1900]
    gdf.loc[gdf['TRACTCE_x'].isin(specified_tracts), 'power'] = 1

    walk_ind_dict = gdf.set_index("GEOID")["NatWalkInd"]
    gross_pop_density_dict = gdf.set_index("GEOID")["D1B"]
    tot_pop_dict = gdf.set_index("GEOID")["TotPop"]
    power_outage = gdf.set_index("GEOID")["power"]

    return walk_ind_dict, gross_pop_density_dict, tot_pop_dict, power_outage, gdf


def get_tower_coordinates(tower_df, county):
    # Collect Tower Data
    print(county)
    # import pdb 
    # pdb.set_trace()
    tower_coordinates = np.array(tower_df[tower_df['county'].isin(county)].apply(lambda x: (x['Longitude'], x['Latitude']), axis=1))
    tower_coordinates = convert_list_to_gdf(tower_coordinates)
    
    return tower_coordinates


def get_score_data():
    # input_file = 'data/final_dist_df_9_counties.csv'
    score_data = pd.read_csv('./data/scores.csv')
    return score_data



im = Image.open('./data/logo.jpeg')
st.set_page_config(layout="wide", page_title='IEEE MOVE Dashboard', page_icon=im, initial_sidebar_state='collapsed')



# SECTION 1: Format the Dashboard look - Title, background color etc..


# Function to set background image for sidebar
def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
side_bg = './data/bg9.jpg'
# sidebar_bg(side_bg)


# Function to set background image for main dashboard
def set_bg_hack_url(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      .stApp  {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
set_bg_hack_url('./data/bg12.jpg')   


st.markdown(
    """<h1 id="logo" style='text-align: center; color: #404040; padding-left:-2550px;'><img src="./app/static/TAMU_logo.svg.png" alt="logo" width="50"/>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MOVE-IT Dashboard  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="./app/static/IEEE_logo.svg.png" alt="logo" width="100"/></h1>""",
    unsafe_allow_html=True)

st.sidebar.title(" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Customize Hotspots")

county = st.multiselect(
   "Select a county",
   ("Brazos", "Anderson"),
   default='Brazos',
#    index=None,
   placeholder="Brazos",
)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 2 - READ AND PREPROCESS ALL DATA

print("[INFO] TIME TAKEN FOR BASIC SETUP --- %.2f seconds ---" % (get_delay()))

### READ ALL INPUT DATA
restaurants, hospitals, fire_stations, parkings, tower_df = get_feature_data()
score_data = get_score_data()
tower_coordinates = get_tower_coordinates(tower_df, county)


print("[INFO] TIME TAKEN FOR READING DATA V1 --- %.2f seconds ---" % ( get_delay()))

walk_ind_dict, gross_pop_density_dict, tot_pop_dict, power_outage, gdf = get_gdf_data()


print("[INFO] TIME TAKEN FOR READING DATA V2  --- %.2f seconds ---" % ( get_delay()))
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 3 - Build Map  


# # Check if center and zoom are in session state, otherwise use default values
# if 'center' not in st.session_state:
#     st.session_state.center = [37.7749, -122.4194]  # Default center (San Francisco)
# if 'zoom' not in st.session_state:
#     st.session_state.zoom = 12  # Default zoom level



m = folium.Map([31.507878, -95.313537], zoom_start=8,  prefer_canvas=True, tiles=None)
folium.raster_layers.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", 
                               attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community', 
                               name='Satellite View',
                            #    opacity=0.7,
                               ).add_to(m)
folium.raster_layers.TileLayer(tiles="openstreetmap", name='Open Street Map').add_to(m)

# # import pdb 
# # pdb.set_trace()

# ## ADD FEATURES TO MAP

# FEATURE 1 - Walkability Index
colormap_walkind = linear.YlGn_09.scale(
    gdf.NatWalkInd.min(), gdf.NatWalkInd.max()
)
folium.GeoJson(
    gdf,
    name='<span style="color:#34568B">FEATURE - Walkability Index</span>',
    style_function=lambda feature: {
        "fillColor": colormap_walkind(walk_ind_dict[feature["properties"]["GEOID"]]),
        "color": "black",
        "weight": 1,
        "dashArray": "5, 5",
        "fillOpacity": 0.5,
    }, 
    show = False,
).add_to(m)


# FEATURE 2- Total Population
colormap_totpop = branca.colormap.LinearColormap(colors=['white', 'yellow', 'orange', 'red'],
                                          index = np.round(np.linspace(gdf.TotPop.min(), gdf.TotPop.max()/5, 4)),
                                          vmin = gdf.TotPop.min(), vmax = gdf.TotPop.max(), tick_labels = np.round(np.exp(np.linspace(gdf.TotPop.min(), gdf.TotPop.max(), 4)),1)
           )
folium.GeoJson(
    gdf,
    name='<span style="color:#34568B">FEATURE - Total Population</span>',
    style_function=lambda feature: {
        "fillColor": colormap_totpop(tot_pop_dict[feature["properties"]["GEOID"]]),
        "color": "black",
        "weight": 1,
        "dashArray": "5, 5",
        "fillOpacity": 0.5,
    }, show = False
).add_to(m)

# FEATURE 3- Tower Locations
towerCluster = MarkerCluster(name='<span style="color:#34568B">FEATURE - Cell Towers</span>', show=False).add_to(m)
tower_locations = [(x.y, x.x) for x in  tower_coordinates['geometry'].sample(1000)]
for coordinates in tower_locations:
    folium.Marker(location = coordinates,
                    icon=BeautifyIcon(icon="phone", 
                                      icon_size=(30,30),
                                     inner_icon_style="font-size:15px;padding-top:1px;",
    #                     #   border_color=color, 
    #                     #     text_color=color, 
                            icon_shape="circle")
                    ).add_to(towerCluster)

# FEATURE 4- Restaurants
restaurant_locations = [(x.y, x.x) for x in  restaurants['geometry'].sample(100)]
restaurantCluster = MarkerCluster(name='<span style="color:#34568B">FEATURE - Restaurants</span>', show=False).add_to(m)
for coordinates in restaurant_locations:
    folium.Marker(location = coordinates,
                    icon=BeautifyIcon(icon="cutlery", 
                    border_color="orange", 
                    icon_size=(30,30),
                    inner_icon_style="font-size:15px;padding-top:1px;",
    #               text_color=color, 
                            icon_shape="circle")
                    ).add_to(restaurantCluster)

# FEATURE 5- Parking
parking_locations = [(x.y, x.x) for x in  parkings['geometry']]
parkingCluster = MarkerCluster(name='<span style="color:#34568B">FEATURE - Parking</span>', show=False).add_to(m)
for coordinates in parking_locations:
    folium.Marker(location = coordinates,
                    icon=BeautifyIcon(icon="car", 
                    border_color="blue", 
                    icon_size=(30,30),
                    inner_icon_style="font-size:15px;padding-top:1px;",
                  text_color="black", 
                            icon_shape="circle")
                    ).add_to(parkingCluster)

# FEATURE 6- Fire Stations
fire_station_locations = [(x.y, x.x) for x in  fire_stations['geometry']]
fire_station_Cluster = MarkerCluster(name='<span style="color:#34568B">FEATURE - Fire Stations</span>', show=False).add_to(m)
for coordinates in fire_station_locations:
    folium.Marker(location = coordinates,
                    icon=BeautifyIcon(icon="fire-extinguisher", 
                    border_color="red", 
                    icon_size=(30,30),
                    inner_icon_style="font-size:15px;padding-top:1px;",
    #               text_color=color, 
                            icon_shape="circle")
                    ).add_to(fire_station_Cluster)

# FEATURE 7- Hospitals  
hospital_locations = [(x.y, x.x) for x in  hospitals['geometry']]
hospital_Cluster = MarkerCluster(name='<span style="color:#34568B">FEATURE - Hospitals</span>', show=False).add_to(m)
for coordinates in hospital_locations:
    folium.Marker(location = coordinates,
                    icon=BeautifyIcon(icon="medkit", 
                    border_color="green", 
                    icon_size=(30,30),
                    text_size = (10,10),
                    inner_icon_style="font-size:15px;padding-top:1px;",
    #               text_color=color, 
                            icon_shape="circle")
                    ).add_to(hospital_Cluster)
    

# -------------------------------------------------------

# INPUT 1 - Power
# Choose discrete colors for 0 and 1
colormap_power = linear.YlGn_09.scale(0,1)
# TO UNDO 
folium.GeoJson(
    gdf[gdf['power']==1],
    name='<span style="color:#BC243C">INPUT - POWER</span>',
    style_function=lambda feature: {
        "fillColor": colormap_power(power_outage[feature['properties']['GEOID']]),
        "color": "blue",
        "weight": 1,
        "dashArray": "5, 5",
        "fillOpacity": 0.5,
    },
    # tooltip=tooltip,
    show = False,
).add_to(m)


# -------------------------------------------------------

## ADD OUTPUTS TO MAP
score_data['boxcox_score'], _ = boxcox(score_data['score'] + 1)  # Adding 1 to avoid issues with zero values
score_data[['longitude', 'latitude']] = score_data['Points'].str.strip('()').str.split(', ', expand=True)
# Convert the new columns to numeric
score_data['latitude'], score_data['longitude']  = pd.to_numeric(score_data['latitude']), pd.to_numeric(score_data['longitude'])
lats, lons = score_data['latitude'], score_data['longitude']


# OUTPUT 1 - Score Points
icon_create_function = """\
function(cluster) {
    return L.divIcon({
    html: '<b>' + cluster.getChildCount() + '</b>',
    className: 'marker-cluster marker-cluster-large',
    iconSize: new L.Point(20, 20)
    });
}"""
# Create a list of tuples with (latitude, longitude)
locations = list(zip(lats, lons))
# marker_cluster = MarkerCluster(
#     locations=locations,    name='<span style="color:green">OUTPUT - Score Points</span>',    overlay=True,
#     control=True,     icon_create_function=icon_create_function,
#     show=False )
# for i, row in score_data.iterrows():
#     tooltip = f"Score: {row['boxcox_score']}"
#     folium.Marker(
#         location=(row['latitude'], row['longitude']),
#         tooltip=tooltip,
#         icon=None,  
#     ).add_to(marker_cluster)
# marker_cluster.add_to(m)


# OUTPUT 2 - Score Heatmap
score_data = score_data.sample(50)
# Extract latitude, longitude, and score from your DataFrame
locations = score_data[['latitude', 'longitude', 'boxcox_score']].values
# Create a HeatMap layer with the color gradient
HeatMap(locations, name= '<span style="color:green">OUTPUT - Score Heatmap</span>',show= False).add_to(m)
# hmap = HeatMap(locations, name= '<span style="color:green">OUTPUT - Score Heatmap</span>',show= False)
# hmap.add_to(m)
# fg = folium.FeatureGroup(name="Main Fea")
# fg.add_child(hmap)
# fg.add_to(m)

# Sample format to use Tooltip
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

print("[INFO] TIME TAKEN FOR DRAWING MAP --- %.2f seconds ---" % (get_delay()))

# Add Layer controls to map
folium.LayerControl().add_to(m)
folium.map.LayerControl('topleft', collapsed= False).add_to(m)
# Add Full screen option
Fullscreen(position="topleft").add_to(m)
# Add map to Dashboard
abc = st_folium(m, height=1050,width=2800, returned_objects=[])
# center=st.session_state["center"], zoom=st.session_state["zoom"]
# map_data = st_folium(m, height=1050,width=2800, feature_group_to_add=fg)
# import pdb
# pdb.set_trace()

print("[INFO] TIME TAKEN TO COMPLETE --- %.2f seconds ---\n\n" % (get_delay()))