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

texas_counties = [
    "Anderson", "Andrews", "Angelina", "Aransas", "Archer", "Armstrong", "Atascosa",
    "Austin", "Bailey", "Bandera", "Bastrop", "Baylor", "Bee", "Bell", "Bexar", "Blanco",
    "Borden", "Bosque", "Bowie", "Brazoria", "Brazos", "Brewster", "Briscoe", "Brooks",
    "Brown", "Burleson", "Burnet", "Caldwell", "Calhoun", "Callahan", "Cameron", "Camp",
    "Carson", "Cass", "Castro", "Chambers", "Cherokee", "Childress", "Clay", "Cochran",
    "Coke", "Coleman", "Collin", "Collingsworth", "Colorado", "Comal", "Comanche", "Concho",
    "Cooke", "Coryell", "Cottle", "Crane", "Crockett", "Crosby", "Culberson", "Dallam",
    "Dallas", "Dawson", "Deaf Smith", "Delta", "Denton", "DeWitt", "Dickens", "Dimmit",
    "Donley", "Duval", "Eastland", "Ector", "Edwards", "El Paso", "Ellis", "Erath", "Falls",
    "Fannin", "Fayette", "Fisher", "Floyd", "Foard", "Fort Bend", "Franklin", "Freestone",
    "Frio", "Gaines", "Galveston", "Garza", "Gillespie", "Glasscock", "Goliad", "Gonzales",
    "Gray", "Grayson", "Gregg", "Grimes", "Guadalupe", "Hale", "Hall", "Hamilton", "Hansford",
    "Hardeman", "Hardin", "Harris", "Harrison", "Hartley", "Haskell", "Hays", "Hemphill",
    "Henderson", "Hidalgo", "Hill", "Hockley", "Hood", "Hopkins", "Houston", "Howard",
    "Hudspeth", "Hunt", "Hutchinson", "Irion", "Jack", "Jackson", "Jasper", "Jeff Davis",
    "Jefferson", "Jim Hogg", "Jim Wells", "Johnson", "Jones", "Karnes", "Kaufman", "Kendall",
    "Kenedy", "Kent", "Kerr", "Kimble", "King", "Kinney", "Kleberg", "Knox", "Lamar", "Lamb",
    "Lampasas", "La Salle", "Lavaca", "Lee", "Leon", "Liberty", "Limestone", "Lipscomb",
    "Live Oak", "Llano", "Loving", "Lubbock", "Lynn", "McCulloch", "McLennan", "McMullen",
    "Madison", "Marion", "Martin", "Mason", "Matagorda", "Maverick", "Medina", "Menard",
    "Midland", "Milam", "Mills", "Mitchell", "Montague", "Montgomery", "Moore", "Morris",
    "Motley", "Nacogdoches", "Navarro", "Newton", "Nolan", "Nueces", "Ochiltree", "Oldham",
    "Orange", "Palo Pinto", "Panola", "Parker", "Parmer", "Pecos", "Polk", "Potter", "Presidio",
    "Rains", "Randall", "Reagan", "Real", "Red River", "Reeves", "Refugio", "Roberts",
    "Robertson", "Rockwall", "Runnels", "Rusk", "Sabine", "San Augustine", "San Jacinto",
    "San Patricio", "San Saba", "Schleicher", "Scurry", "Shackelford", "Shelby", "Sherman",
    "Smith", "Somervell", "Starr", "Stephens", "Sterling", "Stonewall", "Sutton", "Swisher",
    "Tarrant", "Taylor", "Terrell", "Terry", "Throckmorton", "Titus", "Tom Green", "Travis",
    "Trinity", "Tyler", "Upshur", "Upton", "Uvalde", "Val Verde", "Van Zandt", "Victoria",
    "Walker", "Waller", "Ward", "Washington", "Webb", "Wharton", "Wheeler", "Wichita",
    "Wilbarger", "Willacy", "Williamson", "Wilson", "Winkler", "Wise", "Wood", "Yoakum",
    "Young", "Zapata", "Zavala"
]

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
    amenities_file_name = 'data/processed_data/amenities.csv'
    # amenities_data_file = 'data/amenities_9_counties.pkl'

    final_points_data_file = 'data/final_points.pkl'
    
    tower_df = pd.read_csv(tower_data_file)

    amenities_df = pd.read_csv(amenities_file_name, index_col=0)

    

    # restaurants = convert_list_to_gdf(amenities_dict['Food_locations'])
    # hospitals = convert_list_to_gdf(amenities_dict['Hospital_locations'])
    # fire_stations = convert_list_to_gdf(amenities_dict['Fire_station_locations'])
    # parkings = convert_list_to_gdf(amenities_dict['Car_parking_locations'])
    restaurant_df = amenities_df[amenities_df['amenity_type']=='Restaurant']
    restaurants = convert_list_to_gdf(restaurant_df['coordinates'])
    restaurants['county'] = restaurant_df['county'].to_list()

    hospital_df = amenities_df[amenities_df['amenity_type']=='Hospital']
    hospitals = convert_list_to_gdf(hospital_df['coordinates'].to_list())
    hospitals['county'] = hospital_df['county'].to_list()

    fire_stations_df = amenities_df[amenities_df['amenity_type']=='Fire_station']
    fire_stations = convert_list_to_gdf(fire_stations_df['coordinates'].to_list())
    fire_stations['county'] = fire_stations_df['county'].to_list()

    parkings_df = amenities_df[amenities_df['amenity_type']=='Parking']
    parkings = convert_list_to_gdf(parkings_df['coordinates'].to_list())
    parkings['county'] = parkings_df['county'].to_list()
    

    return restaurants, hospitals, fire_stations, parkings, tower_df #tower_coordinates


@st.cache_data
def get_gdf_data():
    # Set filepath
    # fp = "./data/filtered_gdf_v2.csv"
    fp = "data/processed_data/census_data_bg.csv"
    # Read file using gpd.read_file()
    gdf_read = gpd.read_file(fp, 
                        GEOM_POSSIBLE_NAMES="geometry", 
                        KEEP_GEOM_COLUMNS="NO") 

    print("[INFO] TIME TAKEN FOR Reading Census Data --- %.2f seconds ---" % (get_delay()))
    gdf_read['GEOID_12'] = gdf_read['GEOID_12'].apply(lambda x:int(x))
    gdf_read['TotPop'] = gdf_read['TotPop'].apply(lambda x:int(x))
    gdf_read['D1B'] = gdf_read['D1B'].apply(lambda x:float(x))
    gdf_read['NatWalkInd'] = gdf_read['NatWalkInd'].apply(lambda x:float(x))
    gdf_read['STATEFP_x'] = gdf_read['STATEFP_x'].astype('int64')
    gdf_read['COUNTYFP_x'] = gdf_read['COUNTYFP_x'].astype('int64')
    gdf_read['TRACTCE_x'] = gdf_read['TRACTCE_x'].astype('int64')
    gdf_read['BLKGRPCE_x'] = gdf_read['BLKGRPCE_x'].astype('int64')
    print("[INFO] TIME TAKEN FOR Processing --- %.2f seconds ---" % (get_delay()))

    gdf = gpd.GeoDataFrame(gdf_read[['STATEFP_x', 'COUNTYFP_x', 'TRACTCE_x', 'BLKGRPCE_x',  'GEOID', 'geometry', 'TotPop', 'D1B', 'NatWalkInd','GEOID_12', 'county']])
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




im = Image.open('./data/images/logo.jpeg')
st.set_page_config(layout="wide", page_title='IEEE MOVE Dashboard', page_icon=im, initial_sidebar_state='expanded') # state can be "auto" or "collapsed" or "expanded"

st.markdown("""
        <style>
               .main .block-container {
                    padding-top: 1rem;
                    padding-bottom: 15rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)


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
side_bg = './data/images/bg9.jpg'
# sidebar_bg(side_bg)


# # Function to set background image for main dashboard
# def set_bg_hack_url(side_bg):

#    side_bg_ext = 'png'

#    st.markdown(
#       f"""
#       <style>
#       .stApp  {{
#           background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
#       }}
#       </style>
#       """,
#       unsafe_allow_html=True,
#       )
# set_bg_hack_url('./data/bg12.jpg')   

def set_bg_hack_url(side_bg):
    side_bg_ext = 'png'  # Make sure this matches your image's format

    # Added CSS for background size, position, and overlay
    st.markdown(
        f"""
        <style>
        .stApp  {{
           background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
       }}
        .stApp::before {{
            content: '';
            display: block;
            position: absolute;
            top: 0; right: 0; bottom: 0; left: 0;
            background: rgba(255, 255, 255, 0.5);  # Adjust color and opacity for overlay
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_bg_hack_url('./data/images/bg12.jpg')


st.markdown(
    """
    <h1 style='text-align: center; color: #333333; font-family: "Helvetica", Arial, sans-serif;'>
        <img src="./app/static/TAMU_logo.svg.png" alt="TAMU logo" width="50" style="vertical-align: left;" />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  MOVE-IT Dashboard &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
        <img src="./app/static/IEEE_logo.svg.png" alt="IEEE logo" width="80" style="vertical-align: right;" />
    </h1>
    <hr style='border-top: 1px solid #dddddd; margin-top: 10px; margin-bottom: 10px;'/>
    """,
    unsafe_allow_html=True
)

st.write("\n")
st.write("\n")
st.write("\n")

sidebar = st.sidebar
sidebar.title(" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Customize Hotspots")


dropbox = sidebar.container()
county = dropbox.multiselect(
   "Select a county",
   texas_counties,
   default='Brazos',
#    index=None,
   placeholder="Brazos",
)

sidebar.write("\n")
sidebar.write("\n")
sidebar.write("\n")
sidebar.write("\n")
sidebar.write("\n")
sidebar.write("\n")
sidebar.write("\n")
sidebar.write("\n")
sidebar.write("\n")

sidebar.markdown("""<h1 style='text-align: center; padding-left:-2550px;'> Optimize For Features </h1>""",
                 unsafe_allow_html=True)
sidebar.write("Optimize hotspot based on importance : Use the slider to assign weights to each feature")
sidebar.write("\n")
sidebar.write("\n")
sidebar.write("\n")

sliders = sidebar.container()
# Define sliders in Streamlit for weights
tower_weight = sliders.slider("Tower Weight", 0.0, 1.0, 0.4)
hospital_weight = sliders.slider("Hospital Weight", 0.0, 1.0, 0.2)
food_weight = sliders.slider("Food Facility Weight", 0.0, 1.0, 0.1)
parking_weight = sliders.slider("Parking Weight", 0.0, 1.0, 0.4)
totpop_weight = sliders.slider("Total Population Weight", 0.0, 1.0, 0.4)
natwalkind_weight = sliders.slider("Natural Walkability Index Weight", 0.0, 1.0, 0.3)
# """<h1 style='text-align: center; padding-left:-2550px;'>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MOVE-IT Dashboard  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</h1>"""
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 2 - READ AND PREPROCESS ALL DATA

print("[INFO] TIME TAKEN FOR BASIC SETUP --- %.2f seconds ---" % (get_delay()))

### READ ALL INPUT DATA
restaurants, hospitals, fire_stations, parkings, tower_df = get_feature_data()
restaurants  = restaurants[restaurants['county'].isin(county)]
hospitals  = hospitals[hospitals['county'].isin(county)]
fire_stations  = fire_stations[fire_stations['county'].isin(county)]
parkings  = parkings[parkings['county'].isin(county)]
tower_coordinates = get_tower_coordinates(tower_df, county)


# score_data = pd.read_csv('./data/scores.csv')
score_data = pd.read_csv('data/processed_data/candidate_points.csv')
score_data = score_data[score_data['county'].isin(county)]


print("[INFO] TIME TAKEN FOR READING DATA V1 --- %.2f seconds ---" % ( get_delay()))
walk_ind_dict, gross_pop_density_dict, tot_pop_dict, power_outage, gdf = get_gdf_data()
gdf = gdf[gdf['county'].isin(county)]


def get_score(x):
    # Calculate individual scores and add them as new columns
    x['tower_score'] = (1 - x['min_tower_distance_scaled']) * tower_weight
    x['hospital_score'] = (1 - x['min_hospital_distance_scaled']) * hospital_weight
    x['food_score'] = (1 - x['min_food_distance_scaled']) * food_weight
    x['parking_score'] = (1 - x['min_parking_distance_scaled']) * parking_weight
    x['totpop_score'] = x['TotPop_scaled'] * totpop_weight
    x['natwalkind_score'] = x['NatWalkInd_scaled'] * natwalkind_weight
    
    # Calculate the total weight
    total_weight = tower_weight + hospital_weight + food_weight + parking_weight + totpop_weight + natwalkind_weight
    
    # Calculate and add the total score
    x['score'] = (x['tower_score'] + x['hospital_score'] + x['food_score'] + x['parking_score'] + x['totpop_score'] + x['natwalkind_score']) / total_weight

    return x

score_data = get_score(score_data)
print(score_data)
print(score_data.columns)
print(score_data['score'].max())
print(score_data['score'].min())

print("[INFO] TIME TAKEN FOR READING DATA V2  --- %.2f seconds ---" % ( get_delay()))
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECTION 3 - Build Map  


# # Check if center and zoom are in session state, otherwise use default values
# if 'center' not in st.session_state:
#     st.session_state.center = [37.7749, -122.4194]  # Default center (San Francisco)
# if 'zoom' not in st.session_state:
#     st.session_state.zoom = 12  # Default zoom level



m = folium.Map([31.507878, -95.313537], zoom_start=8,  prefer_canvas=True, tiles=None)
folium.raster_layers.TileLayer(tiles="openstreetmap", name='Open Street Map').add_to(m)
folium.raster_layers.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", 
                               attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community', 
                               name='Satellite View',
                            #    opacity=0.7,
                               ).add_to(m)




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
# m.add_child(colormap_walkind)

# FEATURE 3- Tower Locations
towerCluster = MarkerCluster(name='<span style="color:#34568B">FEATURE - Cell Towers</span>', show=False).add_to(m)
tower_locations = [(x.y, x.x) for x in  tower_coordinates['geometry'].sample(100)]
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
restaurant_locations = [(x.y, x.x) for x in  restaurants['geometry']]
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
# score_data = score_data.sample(100)
score_data = score_data.groupby('Geo12_ID').apply(lambda x: x.nlargest(20, 'score')).reset_index(drop=True)
print(score_data['score'].min())
print(score_data['score'].max())
print(score_data)
print(score_data['score'])
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
marker_cluster = MarkerCluster(
    #locations=locations,    
    name='<span style="color:green">OUTPUT - Score Points</span>',    overlay=True,
    control=True,     icon_create_function=icon_create_function,
    show=True )
for i, row in score_data.iterrows():
    # tooltip = f"Score: {row['boxcox_score']}"
    # print(row)
    tooltip = f"Distance from Cell Tower: <strong>{np.round(row['min_tower_distance'],2)} miles</strong></br> \
        Distance from Restaurants : <strong>{np.round(row['min_food_distance'],2)}  miles</strong></br> \
        Distance from Parking : <strong>{np.round(row['min_parking_distance'],2)}  miles</strong></br> \
        Distance from Hospital : <strong>{np.round(row['min_hospital_distance'],2)}  miles</strong></br> \
        Walkability Index (Blockgroup) : <strong>{np.round(row['NatWalkInd'],2)}</strong></br> \
        Total Population (Blockgroup) : <strong>{np.round(row['TotPop'],2)}</strong></br>"
    folium.Marker(
        location=(row['latitude'], row['longitude']),
        tooltip=tooltip,
        icon=folium.DivIcon(html=('<svg height="200" width="200">'
    '<circle cx="30" cy="30" r="22" stroke="green" stroke-width="5" fill="white" opacity="0.6"/>'
    f"""<text x="13.5" y="35" fill="FCF6AE">{np.round(row['score']*100,1)}%</text>"""
    '</svg>')),  
    ).add_to(marker_cluster)
marker_cluster.add_to(m)


# OUTPUT 2 - Score Heatmap
# score_data = score_data.sample(50)
# Extract latitude, longitude, and score from your DataFrame
# locations = score_data[['latitude', 'longitude', 'boxcox_score']].values
locations = score_data[['latitude', 'longitude']].values
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
abc = st_folium(m, height=900,width=2800, returned_objects=[])
st.markdown("""<h6 style='text-align: center; padding-left:-2550px; color: #404040; '>  Each circle on the map denotes a candidate point with a number indicating its percentage score, reflecting the collective impact of all features.</h6>""", unsafe_allow_html=True)
# st.write("Each circle on the map denotes a candidate point with a number indicating its percentage score, reflecting the collective impact of all features.")
# center=st.session_state["center"], zoom=st.session_state["zoom"]
# map_data = st_folium(m, height=1050,width=2800, feature_group_to_add=fg)
# import pdb
# pdb.set_trace()
# st.markdown(str(colormap_power._repr_html_()).join(colormap_power._repr_html_()), unsafe_allow_html=True)
st.markdown("")
st.markdown("""<h6 style='text-align: center;  color: #404040; margin: 0; padding: 0;'> <b>Gradient Legend </b></h6>""", unsafe_allow_html=True)
st.markdown("<span style='color: #404040; margin: 0; padding: 0;'>&emsp;&emsp;&emsp;&emsp; Walkability Index&emsp;</span>" + colormap_walkind._repr_html_() +
            "<span style='color: #404040; margin: 0; padding: 0;'>&emsp;&emsp;&emsp;&emsp; Population &emsp;</span>" + colormap_totpop._repr_html_() + 
            "<span style='color: #404040; margin: 0; padding: 0;'>&emsp;&emsp;&emsp;&emsp; Power &emsp;</span>" + colormap_power._repr_html_(), unsafe_allow_html=True)


print("[INFO] TIME TAKEN TO COMPLETE --- %.2f seconds ---\n\n" % (get_delay()))