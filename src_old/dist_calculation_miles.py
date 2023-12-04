from random import random
from time import sleep
from tqdm import tqdm 
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
from geopy.distance import distance

from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, cpu_count
from shapely import wkt



tower_coordinates = np.array([])

def get_min_distance(args):

    point, amenities_dict = args    
    tower_distances = [distance((row[1], row[0]), (point[1], point[0])).miles for row in amenities_dict['Tower_locations']]
    hospital_distances = [distance((row[1], row[0]), (point[1], point[0])).miles for row in amenities_dict['Hospital_locations']]
    parking_distances = [distance((row[1], row[0]), (point[1], point[0])).miles for row in amenities_dict['Car_parking_locations']]
    food_distances = [distance((row[1], row[0]), (point[1], point[0])).miles for row in amenities_dict['Food_locations']]

    return np.min(tower_distances), np.min(hospital_distances), np.min(parking_distances), np.min(food_distances)


def main():
    global tower_coordinates, index

    print("[INFO] Started MAIN")
    # tower_df = pd.read_csv('./data/Cellular_Towers_V1.csv')
    # # counties = ["Brazos", "Grimes", "Madison", "Leon", "Robertson", "Milam", "Burleson", "Washington", "Waller"]
    # counties = ["Brazos"]
    # counties = [county.upper() for county in counties]
    # print("[INFO] Read Dataset")
    # filtered_tower_df = tower_df[tower_df.LocCounty.isin(counties)]
    # tower_coordinates = np.array(filtered_tower_df.apply(lambda x: (x['X'], x['Y']), axis=1))

    # Collect Tower Data
    # tower_df = pd.read_csv('data/filtered_tower_df.csv')
    tower_df = pd.read_csv('data/filtered_tower_df_9_counties.csv')
    tower_coordinates = np.array(tower_df.apply(lambda x: (x['Longitude'], x['Latitude']), axis=1))

    # Collect other Amenities Data
    # with open('data/amenities.pkl', 'rb') as handle:
    with open('data/amenities_9_counties.pkl', 'rb') as handle:
        amenities_dict = pickle.load(handle)
    amenities_dict['Tower_locations'] = tower_coordinates

    print(f"Total tower locations : {tower_coordinates.shape[0]}")
    print(f"Total hospital locations : {len(amenities_dict['Hospital_locations'])}")
    print(f"Total food locations : {len(amenities_dict['Food_locations'])}")
    print(f"Total parking locations : {len(amenities_dict['Car_parking_locations'])}")

    # Collect Tent point candidates
    with open('data/final_points.pkl', 'rb') as handle:
        points = pickle.load(handle)
    points = gpd.GeoDataFrame(points)
    points['wkb'] = points.apply(lambda x: wkt.loads(x['geometry']), axis=1)
    points['XY'] = points['wkb'].apply(lambda x: (x.x, x.y))
    points = [i for i in points['XY']]
    print(f"\nTotal candidate points: {len(points)}")

    # Prepare argument list for parallel processing
    args_list = [(point, amenities_dict) for point in points]

    print("[INFO] Beginning Multiprocessing")
    # Use multiprocessing Pool to parallelize the computation
    min_distances = []
    min_tower_distances = []
    min_hospital_distances = []
    min_food_distances = []
    min_parking_distances = []
    with Pool() as pool:
        # execute tasks in order
        for result in list(tqdm(pool.imap(get_min_distance, args_list), total=len(args_list))):
            min_tower_distances.append(result[0])
            min_hospital_distances.append(result[1])
            min_food_distances.append(result[2])
            min_parking_distances.append(result[3])


    # Compile final dataset
    score_df = pd.DataFrame({'Points': points,
                             'min_tower_distance': min_tower_distances,
                             'min_hospital_distance': min_hospital_distances,
                             'min_food_distance': min_food_distances,
                             'min_parking_distance': min_parking_distances})
    
    # Scale  all the distances
    tower_distance_scaler = MinMaxScaler()
    tower_distance_scaler.fit(score_df['min_tower_distance'].values.reshape(-1, 1))
    score_df['min_tower_distance_scaled'] = tower_distance_scaler.transform(score_df['min_tower_distance'].values.reshape(-1, 1))

    hospital_distance_scaler = MinMaxScaler()
    hospital_distance_scaler.fit(score_df['min_hospital_distance'].values.reshape(-1, 1))
    score_df['min_hospital_distance_scaled'] = hospital_distance_scaler.transform(score_df['min_hospital_distance'].values.reshape(-1, 1))

    food_distance_scaler = MinMaxScaler()
    food_distance_scaler.fit(score_df['min_food_distance'].values.reshape(-1, 1))
    score_df['min_food_distance_scaled'] = food_distance_scaler.transform(score_df['min_food_distance'].values.reshape(-1, 1))

    parking_distance_scaler = MinMaxScaler()
    parking_distance_scaler.fit(score_df['min_parking_distance'].values.reshape(-1, 1))
    score_df['min_parking_distance_scaled'] = parking_distance_scaler.transform(score_df['min_parking_distance'].values.reshape(-1, 1))

    tower_weight = 0.3
    hospital_weight = 0.2
    food_weight = 0.1
    parking_weight = 0.4

    # Function to calculate score of each point
    def get_score(x):
        return ((1 - x['min_tower_distance_scaled'])*tower_weight + 
                (1 - x['min_hospital_distance_scaled'])*hospital_weight + 
                (1 - x['min_food_distance_scaled'])*food_weight + 
                (1 - x['min_parking_distance_scaled'])*parking_weight) / (tower_weight + hospital_weight + food_weight + parking_weight)
    
    score_df['score'] = score_df.apply(lambda x: get_score(x), axis=1)

    print(score_df.head())

    # Save the dataframe
    score_df.to_csv('data/final_score_df_9_counties.csv')

if __name__ == "__main__":
    main()