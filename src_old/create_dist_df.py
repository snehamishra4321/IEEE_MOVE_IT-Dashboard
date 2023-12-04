from random import random
from time import sleep
from tqdm import tqdm 
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
from geopy.distance import distance
from scipy.spatial.distance import cdist

from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, cpu_count
from shapely import wkt


def main():

    print("[INFO] Started MAIN")

    tower_data_file = 'data/filtered_tower_df_9_counties.csv'
    amenities_data_file = 'data/amenities_9_counties.pkl'
    final_points_data_file = 'data/final_points.pkl'

    output_file = 'data/final_dist_df_9_counties.csv'


    # Collect Tower Data
    tower_df = pd.read_csv(tower_data_file)
    tower_coordinates = np.array(tower_df.apply(lambda x: (x['Longitude'], x['Latitude']), axis=1))

    # Collect other Amenities Data
    with open(amenities_data_file, 'rb') as handle:
        amenities_dict = pickle.load(handle)
    amenities_dict['Tower_locations'] = tower_coordinates

    print(f"Total tower locations : {tower_coordinates.shape[0]}")
    print(f"Total hospital locations : {len(amenities_dict['Hospital_locations'])}")
    print(f"Total food locations : {len(amenities_dict['Food_locations'])}")
    print(f"Total parking locations : {len(amenities_dict['Car_parking_locations'])}")

    # Collect Tent point candidates
    with open(final_points_data_file, 'rb') as handle:
        points_df = pickle.load(handle)
    points_df = gpd.GeoDataFrame(points_df)
    points_df['wkb'] = points_df.apply(lambda x: wkt.loads(x['geometry']), axis=1)
    points_df['XY'] = points_df['wkb'].apply(lambda x: (x.x, x.y))
    points = [i for i in points_df['XY']]
    print(f"\nTotal candidate points: {len(points)}")

    points_list = np.array([list(item) for item in points])
    tower_coordinates = np.array([list(item) for item in tower_coordinates])
    hospital_coordinates = np.array([list(item) for item in amenities_dict['Hospital_locations']])
    food_coordinates = np.array([list(item) for item in amenities_dict['Food_locations']])
    parking_coordinates = np.array([list(item) for item in amenities_dict['Car_parking_locations']])
    tower_dist = np.min(cdist(points_list, tower_coordinates), axis=-1)
    hospital_dist = np.min(cdist(points_list, hospital_coordinates), axis=-1)
    food_dist = np.min(cdist(points_list, food_coordinates), axis=-1)
    parking_dist = np.min(cdist(points_list, parking_coordinates), axis=-1)


    # Compile final dataset
    dist_df = pd.DataFrame({
                             'Points': points,
                             'Geo12_ID' : points_df['GEOID_12'],
                             'min_tower_distance': tower_dist,
                             'min_hospital_distance': hospital_dist,
                             'min_food_distance': food_dist,
                             'min_parking_distance': parking_dist})
    


    # Scale  all the distances
    tower_distance_scaler = MinMaxScaler()
    tower_distance_scaler.fit(dist_df['min_tower_distance'].values.reshape(-1, 1))
    dist_df['min_tower_distance_scaled'] = tower_distance_scaler.transform(dist_df['min_tower_distance'].values.reshape(-1, 1))

    hospital_distance_scaler = MinMaxScaler()
    hospital_distance_scaler.fit(dist_df['min_hospital_distance'].values.reshape(-1, 1))
    dist_df['min_hospital_distance_scaled'] = hospital_distance_scaler.transform(dist_df['min_hospital_distance'].values.reshape(-1, 1))

    food_distance_scaler = MinMaxScaler()
    food_distance_scaler.fit(dist_df['min_food_distance'].values.reshape(-1, 1))
    dist_df['min_food_distance_scaled'] = food_distance_scaler.transform(dist_df['min_food_distance'].values.reshape(-1, 1))

    parking_distance_scaler = MinMaxScaler()
    parking_distance_scaler.fit(dist_df['min_parking_distance'].values.reshape(-1, 1))
    dist_df['min_parking_distance_scaled'] = parking_distance_scaler.transform(dist_df['min_parking_distance'].values.reshape(-1, 1))

    print(dist_df.head())


    # Save the dataframe
    dist_df.to_csv(output_file)

if __name__ == "__main__":
    main()