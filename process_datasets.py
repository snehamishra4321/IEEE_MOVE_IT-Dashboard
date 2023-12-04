import osmnx as ox
import folium 
import numpy as np 
import pandas as pd 
import geopandas as gpd
import os
import pandas as pd
from shapely.geometry import shape, GeometryCollection, Point
from tqdm import tqdm 
import warnings
warnings.filterwarnings("ignore")

###### CELL TOWER DATA

def process_cell_tower_data():
# Link to dataset - https://public.opendatasoft.com/explore/dataset/us-county-boundaries/export/?flg=en-us&disjunctive.statefp&disjunctive.countyfp&disjunctive.name&disjunctive.namelsad&disjunctive.stusab&disjunctive.state_name&q=Texas

    print("[INFO] Reading county boundaries dataset")
    us_county_boundaries = gpd.read_file('data/raw_data/us-county-boundaries-Texas.geojson')
    us_county_boundaries.head()
    # bounds = [x.bounds for x in us_county_boundaries[us_county_boundaries['name'].isin(counties)]['geometry']]
    bounds = [x.bounds for x in us_county_boundaries[us_county_boundaries['statefp']=='48']['geometry']]

    # top_min_long, top_max_long, top_min_lat, top_max_lat = 100, -150, 35, 25
    top_min_long, top_max_long, top_min_lat, top_max_lat = -98, -98, 31, 31
    for min_long, min_lat, max_long, max_lat in bounds:
        if min_long<top_min_long:
            top_min_long = min_long
        if min_lat<top_min_lat:
            top_min_lat = min_lat
        if max_long>top_max_long:
            top_max_long = max_long
        if max_lat>top_max_lat:
            top_max_lat = max_lat

    def read_csv_files_in_subfolders(directory):
        # Initialize an empty list to store DataFrames
        all_data = []

        # Iterate over all folders within the specified directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Check if the file has a CSV extension
                if file.endswith(".csv"):
                    # Construct the full path to the CSV file
                    csv_path = os.path.join(root, file)
                    
                    # Read the CSV file into a DataFrame and append to the list
                    # columns = ["Radio", "MCC", "MNC", "LAC", "CID", "Longitude", "Latitude", "Range", "Samples", "Changeable_1", "Changeable_2", "Created", "Updated", "AverageSignal"]
                    columns = ["Radio", "MCC", "MNC", "LAC", "CID", "UNK", "Longitude", "Latitude", "Range", "Samples", "Changeable_1", "Created", "Updated", "AverageSignal"]
                    df = pd.read_csv(csv_path, names=columns)
                    all_data.append(df)
                    # all_data[file] = df

        # # Concatenate all DataFrames in the list into a single DataFrame
        result_dataframe = pd.concat(all_data, ignore_index=True)

        return result_dataframe
        # return all_data

    # Specify the directory containing subfolders with CSV files
    directory_path = "data/raw_data/openCellID/"

    # Call the function to read all CSV files within subfolders
    print("[INFO] Reading raw Cell data")
    result_dataframe = read_csv_files_in_subfolders(directory_path)

    min_latitude, max_latitude = top_min_lat, top_max_lat
    min_longitude, max_longitude = top_min_long, top_max_long

    filtered_tower_df = result_dataframe[(result_dataframe['Longitude']>min_longitude) & (result_dataframe['Longitude']<max_longitude) & (
        result_dataframe['Latitude']>min_latitude) & (result_dataframe['Latitude']<max_latitude) ]
    
    filtered_tower_df['geometry'] = filtered_tower_df.apply(lambda x: Point(x['Longitude'], x['Latitude']), axis=1)
  
    filtered_tower_df = gpd.GeoDataFrame(
        filtered_tower_df, geometry='geometry', crs="EPSG:4326"
    )
    filtered_tower_df['county'] = ''
    sindex = filtered_tower_df.sindex

    print("[INFO] Finding counties for each cell tower")
    for geometry, county in tqdm(zip(us_county_boundaries['geometry'], us_county_boundaries['name']), total = us_county_boundaries.shape[0]):
        polygon = geometry
        possible_matches_index = list(sindex.intersection(polygon.bounds))
        possible_matches = filtered_tower_df.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(polygon)]
        # filtered_tower_df.loc[precise_matches.index]['county'] = county
        filtered_tower_df.loc[precise_matches.index, 'county'] = county


    filtered_tower_df.to_csv('data/processed_data/tower_df.csv')



if __name__ == "__main__":
    # print("Hello Hello")
    process_cell_tower_data()