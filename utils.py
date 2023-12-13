from concurrent.futures import ProcessPoolExecutor
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import copy
import math
import numpy as np
import re
import random
import pandas as pd
import json
from tqdm import tqdm
import itertools
from itertools import product
import time
from datetime import datetime
from config import *

def print_solution(manager, routing, solution, data, verbose=False):
    """Prints solution on console.
       And return a dict with data model and solution
    """
    sol = {}
    if verbose:
        print(f'Objective: {solution.ObjectiveValue()}')
    sol["Objective"] = solution.ObjectiveValue()
    # Display dropped nodes.
    dropped_nodes = 'Dropped nodes:'
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if solution.Value(routing.NextVar(node)) == node:
            dropped_nodes += ' {}'.format(manager.IndexToNode(node))
    if verbose:
        print(dropped_nodes)
    sol["dropped_nodes"] = dropped_nodes.split(":")[-1]
    # Print routes
    time_dimension = routing.GetDimensionOrDie('Time')
    distance_dimension = routing.GetDimensionOrDie('Distance')
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    total_time = 0
    total_distance = 0
    total_load = 0
    for vehicle_id in range(manager.GetNumberOfVehicles()):
        index = routing.Start(vehicle_id)
        plan_output = f'Route for vehicle {vehicle_id}:\n'
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            distance_var = distance_dimension.CumulVar(index)
            capacity_var = capacity_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2}) Distance:{3} Load:{4}  -> '.format(
                node_index,
                solution.Min(time_var), solution.Max(time_var),
                solution.Value(distance_var),
                solution.Value(capacity_var))
            index = solution.Value(routing.NextVar(index))
        node_index = manager.IndexToNode(index)
        time_var = time_dimension.CumulVar(index)
        distance_var = distance_dimension.CumulVar(index)
        capacity_var = capacity_dimension.CumulVar(index)
        plan_output += '{0} Time({1},{2}) Distance:{3} Load:{4})\n'.format(
            manager.IndexToNode(index),
            solution.Min(time_var), solution.Max(time_var),
            solution.Value(distance_var),
            solution.Value(capacity_var))
        plan_output += 'Time of the route: {}sec\n'.format(
            solution.Min(time_var))
        plan_output += 'Distance of the route: {}m\n'.format(
            solution.Value(distance_var))
        plan_output += 'Load of the route: {}\n'.format(
            solution.Value(capacity_var))
        sol[vehicle_id] = plan_output
        if verbose:
            print(plan_output)
        total_time += solution.Min(time_var)
        total_distance += solution.Value(distance_var)
        total_load += solution.Value(capacity_var)

    if verbose:
        print('Total time of all routes: {}sec'.format(total_time))
    sol["Total time of all routes"] = total_time
    if verbose:
        print('Total distance of all routes: {}m'.format(total_distance))
    sol["Total distance of all routes"] = total_distance
    if verbose:
        print('Total load of all routes: {}'.format(total_load))
    sol["Total load of all routes"] = total_load

    for i in range(manager.GetNumberOfVehicles()):  # number of vehicule
        data_list = [item.lstrip()
                     for item in sol[i].split("\n")[1].split("->")]

        route_tmp = [re.findall(r'^(\d+)', item)[0] for item in data_list]

        # Extracting the second number from each element
        time_tmp = [re.findall(r'Time\((\d+)', item)[0] for item in data_list]

        # Extracting the value after 'Load:'
        load_values = [re.findall(r'Load:(\d+)', item)[0]
                       for item in data_list]

        sol[i] = {"route": list(map(int, route_tmp)), "time": list(
            map(int, time_tmp)), "load": list(map(int, load_values))}
    return sol


def count_combinations(n, num_combi, cap, verbose):
    if n > 0:
        combinations = findCombinations(n, num_combi, cap, verbose)
        return len(combinations)
    return 1


def model_complexity(dmd, cap=15, verbose=False):
    s = 0
    for n in range(len(dmd)):
        tmpdmd = abs(dmd[n])
        nc = math.ceil(tmpdmd/cap)
        count = count_combinations(tmpdmd, nc, cap, verbose)
        if n == 1:
            s = count
            if verbose:
                print(s, end="")
        else:
            if verbose:
                print(s, "*", count, end="")
            s = s*count
        if verbose:
            print(" = ", s)
    return s


def findCombinationsUtil(arr, index, num, reducedNum, num_combi, cap, combinations):
    if reducedNum < 0:
        return
    if reducedNum == 0 and index == num_combi:
        combi = []
        for i in range(index):
            combi.append(arr[i])
        if max(combi) <= cap:
            combinations.append(combi)
        return
    prev = 1 if index == 0 else arr[index - 1]
    for k in range(prev, min(num + 1, reducedNum + 1)):
        arr[index] = k
        findCombinationsUtil(arr, index + 1, num,
                             reducedNum - k, num_combi, cap, combinations)


def findCombinations(n, num_combi, cap,parquet,verbose=False):
    arr = [0] * n
    combinations = []
    try:
    
        if verbose:
            print("Calculating combinasion")
        parquet = parquet[parquet["cap"] == cap]
        parquet = parquet[parquet["value"] == n]
        if len(parquet) > 0:
            return eval(parquet.combi.iloc[0])
        else:
            findCombinationsUtil(arr, 0, n, n, num_combi, cap, combinations)
            return combinations
    except:
        findCombinationsUtil(arr, 0, n, n, num_combi, cap, combinations)
        return combinations

def get_dataframe(name_file, PATH):
    data_path = fr"{PATH}/Input/{name_file}"
    df = pd.read_csv(data_path, sep=";")
    return df

def base(df):
    '''
    Get GPS positions of all offshore platforms and rigs and create dictionary.
    Add a line to the so called "depot" line to the input dataframe, necessary for ortools.
    Sort to have depot line first (aesthetic of Adam).
    Remove from dictionary, the nodes that do not need to be visited.
    Return GPS positions dict of the nodes and input dataframe.
    '''
    # positon :
    with open('GPS.json', 'r') as f:
    # Load JSON data from file
        GPS = json.load(f)["GPS"]
    dict_positions = GPS
    df.loc[df.index.max() + 1, 'DEMAND'] = 0
    df.loc[df.index.max(), 'DEST_FROM'] = "EBJ0"
    df.loc[df.index.max(), 'DEST_TO'] = "EBJ_start"
    df.loc[df.index.max(), 'BETWEEN_'] = 0
    df.loc[df.index.max(), 'AND_'] = 60 * 60 * 24
    # Convert "DEMAND" column to integer type
    df['DEMAND'] = df['DEMAND'].astype(int)
    df['BETWEEN_'] = df['BETWEEN_'].astype(int)
    df['AND_'] = df['AND_'].astype(int)
    df = df[["DEST_FROM", "DEST_TO", "BETWEEN_", "AND_", "DEMAND"]]
    df = df.sort_values(by="DEMAND", ascending=True)
    keys_to_keep = list(df.DEST_FROM)+list(df.DEST_TO)+["EBJ0"]
    for key in list(dict_positions.keys()):  # use list to copy keys
        if key not in keys_to_keep:
            dict_positions.pop(key)
  
    return df, dict_positions

    
def save_objectif(sol, mode, name,config_singleton=""):
    """ function to explore best meta huristique methode"""
    # Assuming sol is your list of dictionaries
    # Create DataFrame

    try:
        timestamp = [item["timestamp"] for item in sol]
        solution = [item["solution"]["Total distance of all routes"]
                    for item in sol]
        info_values = [{"Objective": item["solution"]["Objective"],
                        "len_demands": len(item["demands"])} for item in sol]
        df_tmp = pd.DataFrame(info_values)

    except:
        timestamp = "nan"
        solution = "nan"
        df_tmp = pd.DataFrame()

    # Add timestamp and solution to the dataframe
    df_tmp["Timestamp"] = timestamp
    df_tmp["Solution"] = solution
    df_tmp["data_model"] = str(sol)


    # Save DataFrame to a Parquet file
    chunk_size = 50000
    chunks = [x for x in range(0, df_tmp.shape[0], chunk_size)]

    for i in range(len(chunks) - 1):
        df_tmp.iloc[chunks[i]:chunks[i + 1]].to_parquet(f"{config_singleton.PATH}/Output/{name}_{mode}_{i}.parquet")

    print("result save")


def test_data_consistency(data):
    """
    This function checks if the data in the given dictionary is consistent.

    Arguments:
    data -- a dictionary containing the data

    The data dictionary is expected to have the following keys:
    'num_vehicles', 'vehicle_capacities', 'time_matrix', 
    'distance_matrix', 'time_windows', 'demands'
    """

    assert data['num_vehicles'] == len(
        data['vehicle_capacities']), "Mismatch in number of vehicles and vehicle capacities."
    assert len(data['time_matrix']) == len(data['distance_matrix']
                                           ), "Mismatch in time matrix and distance matrix sizes."
    assert len(data['time_matrix']) == len(data['time_windows']
                                           ), "Mismatch in time matrix and time windows sizes."
    assert len(data['time_matrix']) == len(data['demands']
                                           ), "Mismatch in time matrix and demands sizes."
