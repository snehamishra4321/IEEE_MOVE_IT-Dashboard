from random import random
from time import sleep
from tqdm import tqdm 
import pickle
import random

import numpy as np
import pandas as pd
import geopandas as gpd
from geopy.distance import distance
from scipy.spatial.distance import cdist

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from multiprocessing import Pool, cpu_count
from shapely import wkt



def main():

    print("[INFO] Started MAIN")

    tower_data_file = 'data/processed_data/tower_df.csv'
    amenities_data_file = 'data/processed_data/amenities.csv'
    final_points_data_file = 'data/final_points.pkl'

    output_file = 'data/processed_data/candidate_points.csv'


    # Collect Tower Data
    tower_df = pd.read_csv(tower_data_file).sample(200000)  # Sampling to avoid memory error
    tower_coordinates = np.array(tower_df.apply(lambda x: (x['Longitude'], x['Latitude']), axis=1))

    def convert_amenities_coordinates(coordinates):
        coordinates = np.array([list(item.replace("(","").replace(")","").split(",")) for item in coordinates])
        return np.array([[float(x[0]), float(x[1])] for x in coordinates])
    

    tower_coordinates = np.array([list(item) for item in tower_coordinates])
    amenities_df = pd.read_csv(amenities_data_file, index_col=0)

    restaurant_df = amenities_df[amenities_df['amenity_type']=='Restaurant']
    food_coordinates = convert_amenities_coordinates(restaurant_df['coordinates'].to_list())

    hospital_df = amenities_df[amenities_df['amenity_type']=='Hospital']
    hospital_coordinates = convert_amenities_coordinates(hospital_df['coordinates'].to_list())

    parkings_df = amenities_df[amenities_df['amenity_type']=='Parking']
    parking_coordinates = convert_amenities_coordinates(parkings_df['coordinates'].to_list())


    # Collect Tent point candidates
    with open(final_points_data_file, 'rb') as handle:
        points_df = pickle.load(handle)
    points_df = gpd.GeoDataFrame(points_df)
    points_df['wkb'] = points_df.apply(lambda x: wkt.loads(x['geometry']), axis=1)
    points_df['XY'] = points_df['wkb'].apply(lambda x: (x.x, x.y))
    points = [i for i in points_df['XY']]
    print(f"\nTotal candidate points: {len(points)}")


    points_list = np.array([list(item) for item in points])
    tower_dist = np.min(cdist(points_list, tower_coordinates), axis=-1)  * 64
    print("Completed Tower Dist Calculation")
    hospital_dist = np.min(cdist(points_list, hospital_coordinates), axis=-1)  * 64
    print("Completed Hospital Dist Calculation")
    food_dist = np.min(cdist(points_list, food_coordinates), axis=-1)  * 64
    print("Completed Food Dist Calculation")
    parking_dist = np.min(cdist(points_list, parking_coordinates), axis=-1)  * 64


    # Compile final dataset
    dist_df = pd.DataFrame({
                             'Points': points,
                             'Geo12_ID' : points_df['GEOID_12'],
                             'min_tower_distance': tower_dist,
                             'min_hospital_distance': hospital_dist,
                             'min_food_distance': food_dist,
                             'min_parking_distance': parking_dist})
    

    def multiple_scalers(series):
        # series = np.log(series)
        scaler_v1 = RobustScaler(quantile_range=(30.0, 70.0))
        transformed_series = scaler_v1.fit_transform(series)
        scaler_v2 = MinMaxScaler()
        transformed_series = scaler_v2.fit_transform(transformed_series)
        return transformed_series

    # Scale  all the distances
    tower_distance_scaler = MinMaxScaler()
    tower_distance_scaler.fit(np.log(dist_df['min_tower_distance'].values).reshape(-1, 1))
    dist_df['min_tower_distance_scaled'] = tower_distance_scaler.transform(np.log(dist_df['min_tower_distance'].values).reshape(-1, 1))
    dist_df['min_hospital_distance_scaled'] = multiple_scalers(dist_df['min_hospital_distance'].values.reshape(-1,1))
    dist_df['min_food_distance_scaled'] = multiple_scalers(dist_df['min_food_distance'].values.reshape(-1,1))
    dist_df['min_parking_distance_scaled'] = multiple_scalers(dist_df['min_parking_distance'].values.reshape(-1,1))

    gdf_v1 = pd.read_csv('data/processed_data/census_data_bg.csv')[['GEOID_12', 'NatWalkInd', 'TotPop', 'county']]
    walkability_df_scaler = MinMaxScaler()
    gdf_v1['NatWalkInd_scaled'] = walkability_df_scaler.fit_transform(gdf_v1['NatWalkInd'].values.reshape(-1,1))
    TotPop_scaler = MinMaxScaler()
    gdf_v1['TotPop_scaled'] = TotPop_scaler.fit_transform(gdf_v1['TotPop'].values.reshape(-1,1))
    final_df = pd.merge(dist_df, gdf_v1, right_on='GEOID_12', left_on = 'Geo12_ID', how='left')

    # Save the dataframe
    final_df.to_csv(output_file)

if __name__ == "__main__":
    main()