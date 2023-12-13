from concurrent.futures import ProcessPoolExecutor
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import copy
import math
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import itertools
from itertools import product
import time
from datetime import datetime
import json
from utils import *
from config import *

def main_data_model(df, meta_cap=15, Slicing=0, num_vehicles=4, soft_limit=10000, verbose=False,config_singleton=""):
    if verbose:
        print("Create Base Data Model")
    if isinstance(Slicing, tf.Tensor) and isinstance(meta_cap, tf.Tensor): 
        Slicing = int(round(Slicing.numpy().item()))
        meta_cap = int(round(meta_cap.numpy().item()))
    df, dict_positions = base(df) # added depot + uncessary nodes removed
    cap_tab_mask = []
    cap_tab = []
    Slicing_tab = []
    if callable(meta_cap):
        for d in df['DEMAND']:
            #lambda pax :random.randint(4,15) if pax >15  else random.randint(4,pax)
            if d < 4: # don't cut
                cap_tab_mask.append(False)
            else:    
                tmp_cap = meta_cap(d)
                
                if d <tmp_cap:
                    cap_tab_mask.append(False)
                else:
                    cap_tab_mask.append(True)
                    cap_tab.append(tmp_cap) 

        df_less_cap = df[np.invert(cap_tab_mask)]
        df_greater_cap = df[cap_tab_mask]
    else:
        df_less_cap = df[df['DEMAND'] <= meta_cap]
        df_greater_cap = df[df['DEMAND'] > meta_cap]
        cap_tab = [meta_cap]*len(df_greater_cap)

    dataframes = []  # list to store all dataframes
    combination_list = []
    
    if verbose:
        for row,cap in zip(tqdm(df_greater_cap.iterrows(), total=len(df_greater_cap), desc="Processing combinations"), cap_tab):
            n = int(row[1]['DEMAND'])
            
            if callable(Slicing):
            #   Slicing = lambda pax,caping : random.randint(math.ceil(pax/caping),pax-math.ceil(pax/caping))
                
                Slicing = Slicing(n,cap)
            Slicing_tab.append(Slicing)
            num_combi = math.ceil(n / cap) + Slicing 
            combinations = findCombinations(n, num_combi, cap,parquet=config_singleton.PARQUET)
            combination_list.append([(row[1], combi) for combi in combinations])
            
        random.shuffle(combination_list)
        stop_wall = 0
        stop_wall_max = 0
        for combo in tqdm(product(*combination_list), desc="Creating all combinations"):
            stop_wall = stop_wall+1
            rows = []
            for row, values in combo:
                for val in values:
                    new_row = row.copy()
                    new_row['DEMAND'] = val
                    # append Series to list
                    rows.append(new_row)

            # convert list of Series to DataFrame
            temp_df = pd.DataFrame(rows, columns=df.columns)
            temp_df = pd.concat([temp_df, df_less_cap])
            dataframes.append(temp_df)
            if stop_wall == soft_limit:
                break
    else:# TODO : FIX VERBOSE = FALSE
        #for row,cap in zip(tqdm(df_greater_cap.iterrows(), total=len(df_greater_cap), desc="Processing combinations"), cap_tab):
        for row,cap in zip(df_greater_cap.iterrows(),cap_tab):
            n = row['DEMAND']
            if callable(Slicing):
            #   Slicing = lambda pax,caping : random.randint(math.ceil(pax/caping),pax-math.ceil(pax/caping))
                Slicing = Slicing(n,cap)
            Slicing_tab.append(Slicing)
            num_combi = math.ceil(n / cap) + Slicing 
            combinations = findCombinations(n, num_combi, cap,parquet=config_singleton.PARQUET)
            combination_list.append([(row, combi) for combi in combinations])

        for combo in product(*combination_list):
            rows = []
            for row, values in combo:
                for val in values:
                    new_row = row.copy()
                    new_row['DEMAND'] = val
                    # append Series to list
                    rows.append(new_row)

            # convert list of Series to DataFrame
            temp_df = pd.DataFrame(rows, columns=df.columns)
            temp_df = pd.concat([temp_df, df_less_cap])
            dataframes.append(temp_df)

    data_models = []

    for df_ind in range(len(dataframes)):
        data = {}

        # num_vehicles

        data["num_vehicles"] = num_vehicles
        # vehicle_capacities
        data["vehicle_capacities"] = copy.copy([15]*num_vehicles)
        data['depot'] = 0
        data["GPS"] = dict_positions

        dataframes[df_ind] = dataframes[df_ind].sort_values(
            by="DEMAND", ascending=True)
        name_demands = list(
            dataframes[df_ind]["DEST_FROM"])+list(dataframes[df_ind]["DEST_TO"])
        data["name_demands"] = name_demands
        demands_index = list(np.arange(len(name_demands)))
        data["demands_index"] = demands_index
        dataframes[df_ind]["DEST_FROM"] = list(
            np.arange(len(list(dataframes[df_ind]["DEST_FROM"]))))
        dataframes[df_ind]["DEST_TO"] = list(np.arange(len(list(dataframes[df_ind]["DEST_FROM"])), len(
            list(dataframes[df_ind]["DEST_FROM"]))+len(list(dataframes[df_ind]["DEST_TO"]))))
        pick_deli = dataframes[df_ind][[
            'DEST_FROM', 'DEST_TO']].values.tolist()
        pick_deli = [sublist for sublist in pick_deli if sublist[0] != 0]
        data["pickups_deliveries"] = pick_deli
        time_windows = dataframes[df_ind][['BETWEEN_', 'AND_']].values.tolist()
        data["time_windows"] = time_windows+time_windows
        demands = list(dataframes[df_ind]["DEMAND"]) + \
            [-i for i in dataframes[df_ind]["DEMAND"]]
        data["demands"] = demands
        data["Cap"] = cap_tab
        data["Slice"] = Slicing_tab

        # Haversine formula to calculate the distance between two lat/lon points on a sphere
        def haversine(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(
                math.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * \
                math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371*1000  # Radius of earth in kilometers
            return c * r

        keys = list(name_demands)
        distance_matrix = []
        data["distance_matrix"] = distance_matrix
        for i in range(len(keys)):
            distance_row = []
            for j in range(len(keys)):
                if i == j:
                    distance_row.append(0)  # distance to self is 0
                else:
                    lat1, lon1 = dict_positions[keys[i]]
                    lat2, lon2 = dict_positions[keys[j]]
                    distance = haversine(lon1, lat1, lon2, lat2)
                    distance_row.append(math.ceil(distance))
            distance_matrix.append(distance_row)
        # time_matrix
        velocity_helico = 62.5  # 225#km/h
        time_matrix = [[math.ceil(distance / velocity_helico)
                        for distance in row] for row in distance_matrix]
        data["time_matrix"] = time_matrix
        data_models.append(data)

    return data_models

def make_dataframe_ready(df):
    '''
    This function converts dataframe datetime columns to seconds.
    '''
    Number_of_hours_a_heli_can_fly = 10
    df['BETWEEN_'] = pd.to_datetime(df['BETWEEN_'])
    df['AND_'] = pd.to_datetime(df['AND_'])

    df['BETWEEN_'] = (df['BETWEEN_'].dt.hour * 3600) +(df['BETWEEN_'].dt.minute * 60) + df['BETWEEN_'].dt.second 
    df['AND_'] = (df['AND_'].dt.hour * 3600) + (df['AND_'].dt.minute * 60) + df['AND_'].dt.second

    #make sure And_ are not too short
    df.loc[df['AND_'] - df['BETWEEN_'] < 3600*Number_of_hours_a_heli_can_fly, 'AND_'] = df['BETWEEN_'] + 3600*Number_of_hours_a_heli_can_fly
    return df




#### Preprocessing NN

import itertools
import json
import numpy as np
import os
import random
import pandas as pd



'''
todo : 
    normalize input


'''

def generate_NN_input(config_singleton = ""):
    def trim_gps_keys(keys):
        return keys[:3]

    gps_keys = config_singleton.GPS.keys()
    trimmed_keys = map(trim_gps_keys, gps_keys)
    # Create a new dictionary to store the combinations
    GPS_combination_dict = {}
    index = 1
    # Generate all possible pairs (combinations of two) of keys
    for combo in itertools.combinations(trimmed_keys, 2):
        GPS_combination_dict[combo] = index
        index += 1  

    files_name = os.listdir(f'{config_singleton.PATH}/Input')
    random.shuffle(files_name)
    datasets = []
    file_names_ = []
    for f in files_name:
        f_n = f'{config_singleton.PATH}/Input/{f}'
        df = pd.read_csv(f_n,sep=";")
        
        v,n= [],[]
        for ind,row in df.iterrows():
            n.append(row[2])
            try:
                v.append(GPS_combination_dict[(row[-2][:3],row[-1][:3])])
            except:
                v.append(GPS_combination_dict[(row[-1][:3],row[-2][:3])])
        while len(n) < 75:
            n.append(0)
            v.append(0)
        nv = np.array(n+v)
        datasets.append(nv)
        file_names_.append(f)
    return datasets,file_names_