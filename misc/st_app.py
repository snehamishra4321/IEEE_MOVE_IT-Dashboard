import random
import math
import pandas as pd
import folium
import numpy as np
import geopandas 
import requests
import branca
# Example usage:
min_latitude = 30.0
max_latitude = 40.0
min_longitude = -120.0
max_longitude = -100.0
max_dist = 200
crowd_fraction = 0.8
crowd_centers = [(35.0, -115.0), (37.5, -110.0), (32.0, -105.0)]  # Example crowd centers
num_points = 500



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

import pandas as pd

# Your code for generating random points here...

# Create a DataFrame from the generated random points
data_points = pd.DataFrame(data_points, columns=['Latitude', 'Longitude'])

# Now, you have a DataFrame containing the generated data
data_points


truckdata = {
    'Longitude': [-96.8115, -100, -110, -95],
    'Latitude': [32.5202, 60, 50, 55]
}

truck_points = pd.DataFrame(truckdata)

# Print the DataFrame
truck_points



data = requests.get(
    "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json"
).json()
states = geopandas.GeoDataFrame.from_features(data, crs="EPSG:4326")

states['strength'] = np.random.randint(0, 101, size=len(states))
states.head()


tower_df = pd.read_csv('./data/celltowers.csv')


from folium.plugins import BeautifyIcon


# Create a map centered around the first location in your data
m = folium.Map(center=[data_points['Latitude'].iloc[0], data_points['Longitude'].iloc[0]], 
               zoom_start=6,
               tiles="OpenStreetMap"#,
            #    control_scale =True, 
            #    zoom=6
               )

# Add markers for each data point
data = folium.FeatureGroup().add_to(m)
for index, row in data_points.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']]).add_to(data)


truck = folium.FeatureGroup().add_to(m)
for index, row in truck_points.iterrows():
    folium.Marker(location = [row['Latitude'], row['Longitude']],icon = folium.plugins.BeautifyIcon(
    icon="truck", border_color="#b3334f", text_color="#b3334f", icon_shape="triangle"
)).add_to(truck)
    

tower = folium.FeatureGroup().add_to(m)
for index, row in tower_df.sample(100).iterrows():
    folium.Marker(location = [row['latitude'], row['longitude']],icon = BeautifyIcon(
    icon="tower-observation", border_color="#b3334f", text_color="#b3334f", icon_shape="triangle"
)).add_to(tower)
    

folium.TileLayer("Stamen Watercolor").add_to(m)
folium.TileLayer("CartoDB Positron", show=False).add_to(m)

#folium.LatLngPopup().add_to(m)
m.fit_bounds(m.get_bounds())
folium.LayerControl().add_to(m)

# Display the map
# m.save('map.html')  # Save the map to an HTML file
# m



from streamlit_folium import st_folium

st_data = st_folium(m, width=725)