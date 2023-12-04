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


im = Image.open('./data/logo.jpeg')
st.set_page_config(layout="wide", page_title='IEEE MOVE Dashboard', page_icon=im, initial_sidebar_state='collapsed')



# SECTION 1: Format the Dashboard look - Title, background color etc..

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

if "selected_id" not in st.session_state:
        st.session_state.selected_id = None
        print("Initialized Session ID")

st.markdown(
    # """<h1 style='text-align: center; color: #404040;'> <img src=> MOVE-IT Dashboard</h1>""",
    """<h1 id="logo" style='text-align: center; color: #404040; padding-left:-2550px;'><img src="./app/static/TAMU_logo.svg.png" alt="logo" width="50"/>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MOVE-IT Dashboard  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="./app/static/IEEE_logo.svg.png" alt="logo" width="100"/></h1>""",
    unsafe_allow_html=True)

sidebar = st.sidebar
sidebar.title(" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Customize Hotspots")
sidebar.text("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
infobox = sidebar.container()
# infobox.write(f"Score of selected point : {st.session_state.selected_id}")

st.markdown(f"""
            <style>
            [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"]   {{ border: 5px groove grey; padding-left: 20px; background-color: #A8A8A8;
            }}
            </style>
            """,
      unsafe_allow_html=True,
      )


@st.cache_resource  # @st.cache_data
def load_map():
    # Load the map
    m = folium.Map([31.507878, -95.313537], zoom_start=8,  prefer_canvas=True, tiles=None)
    folium.raster_layers.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", 
                                attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community', 
                                name='Satellite View',
                                #    opacity=0.7,
                                ).add_to(m)
    folium.raster_layers.TileLayer(tiles="openstreetmap", name='Open Street Map').add_to(m)

    def get_score_data():
        # input_file = 'data/final_dist_df_9_counties.csv'
        score_data = pd.read_csv('./data/scores.csv')
        return score_data
    score_data = get_score_data()
    score_data['boxcox_score'], _ = boxcox(score_data['score'] + 1)  # Adding 1 to avoid issues with zero values
    score_data[['longitude', 'latitude']] = score_data['Points'].str.strip('()').str.split(', ', expand=True)
    # Convert the new columns to numeric
    score_data['latitude'], score_data['longitude']  = pd.to_numeric(score_data['latitude']), pd.to_numeric(score_data['longitude'])
    lats, lons = score_data['latitude'], score_data['longitude']
    # OUTPUT 2 - Score Heatmap
    score_data = score_data.sample(50)
    # Extract latitude, longitude, and score from your DataFrame
    locations = score_data[['latitude', 'longitude', 'boxcox_score']].values
    # Create a HeatMap layer with the color gradient
    # HeatMap(locations, name= '<span style="color:green">OUTPUT - Score Heatmap</span>',show= False).add_to(m)
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
    marker_cluster = MarkerCluster(
        name='<span style="color:green">OUTPUT - Score Points</span>',    overlay=True,
        control=True,     icon_create_function=icon_create_function,
        show=True )
    for i, row in score_data.iterrows():
        tooltip = f"Score: {row['boxcox_score']}"
        

        folium.Marker(
            location=(row['latitude'], row['longitude']),
            # tooltip=tooltip,
            tooltip=f"ID: {i}",
            icon=None,  
        ).add_to(marker_cluster)
        # ).add_to(m)
    marker_cluster.add_to(m)
    # df = load_df()  # load data
    # m = plot_from_df(df, m)  # plot points
    # folium.LayerControl().add_to(m)
    folium.map.LayerControl('topleft', collapsed= False).add_to(m)
    # Add Full screen option
    Fullscreen(position="topleft").add_to(m)

    return m, score_data

m, score_data = load_map()
print(score_data.columns)


print("[INFO] TIME TAKEN FOR DRAWING MAP --- %.2f seconds ---" % (get_delay()))

# Add Layer controls to map
# folium.LayerControl().add_to(m)
# folium.map.LayerControl('topleft', collapsed= False).add_to(m)
# Add Full screen option
# Fullscreen(position="topleft").add_to(m)
# Add map to Dashboard
map_data = st_folium(m, height=1050,width=2800)
st.session_state.selected_id = map_data['last_object_clicked_tooltip']

print(map_data)
# print(st.session_state.selected_id)

# import pdb
# pdb.set_trace()
if st.session_state.selected_id is not None:
    print(st.session_state.selected_id)
    index = int(st.session_state.selected_id.split(":")[1])
    infobox.write(f"ID of selected point : {st.session_state.selected_id}")
    infobox.write(f"Score of selected point : {np.round(score_data.loc[index]['boxcox_score'],2)}")
    infobox.write(f"Distance to closest Cell Tower : {np.round(score_data.loc[index]['min_tower_distance'],2)}")
    infobox.write(f"Distance to closest Hospital   : {np.round(score_data.loc[index]['min_hospital_distance'],2)}")
    infobox.write(f"Distance to closest Parking    : {np.round(score_data.loc[index]['min_parking_distance'],2)}")
    infobox.write(f"Distance to closest Restaurant : {np.round(score_data.loc[index]['min_food_distance'],2)}")
    infobox.write(f"Walkability Index (BlockGroup) : {np.round(score_data.loc[index]['NatWalkInd'],2)}")
    infobox.write(f"Total Population (BlockGroup)  : {np.round(score_data.loc[index]['TotPop'],2)}")
    
# center=st.session_state["center"], zoom=st.session_state["zoom"]
# map_data = st_folium(m, height=1050,width=2800, feature_group_to_add=fg)
# import pdb
# pdb.set_trace()

print("[INFO] TIME TAKEN TO COMPLETE --- %.2f seconds ---\n\n" % (get_delay()))