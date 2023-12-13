from concurrent.futures import ProcessPoolExecutor
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import copy
import math
import numpy as np
import re
import random
import os
import pandas as pd
from tqdm import tqdm
import itertools
from itertools import product
import time
from datetime import datetime
import json
from solver import solve_vrp
from utils import *
from preprocessing import main_data_model,make_dataframe_ready
from heuristic import meta_herusitic_data_model_processor

def main(data_models, max_workers=1, verbose=False,config_singleton=""):
        
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        data_list = list(enumerate(data_models))
        data_models_solved = list(executor.map(solve_vrp, data_list, [len(
            data_models)]*len(data_models), [verbose]*len(data_models), [config_singleton]*len(data_models)))
    return data_models_solved
def solve(df, caping, sclicing, num_vehicles, soft_limit, file_name,prefix_name="" ,verbose=True,list_mode= ["random_brute_force"],max_workers=1,save=True,config_singleton=""):
        print("doing model")
        print(caping, sclicing)
        data_models = main_data_model(df, caping, sclicing, num_vehicles, soft_limit, verbose,config_singleton=config_singleton)
        mode = list_mode[0]
        data_models_deep_copy = copy.copy(data_models)
        print(f"Sorting with meta heuristic mode : {mode}")
        data_models_copy = meta_herusitic_data_model_processor(
                data_models_deep_copy, mode, 100)
        print("Starting solver ...")
        sol = main(data_models_copy, max_workers=max_workers, verbose=verbose)
        if save:
            if callable(caping):
                caping="random"
                sclicing = "random"
            save_objectif(sol, mode, f"{file_name[:-4]}_{str(caping)}_{str(sclicing)}_{prefix_name}",config_singleton=config_singleton)
        return sol
def calculating_plus_n_pax(df, caping, sclicing, num_vehicles, soft_limit, file_name,max_workers,prefix_name="",n=1):
        
        for i in range(len(df)):
            try:
                df.DEMAND.iloc[i]  = df.DEMAND.iloc[i]+1 
                solve(df, caping, sclicing, num_vehicles, soft_limit, file_name,prefix_name=str(df["DEST_FROM"].iloc[i]+"_1"),max_workers=max_workers)
                df.DEMAND.iloc[i]  = df.DEMAND.iloc[i]-1 

            except:
                print("Fail, going next")

#pre_model_search_mode
def full_random(df,  num_vehicles, soft_limit, file_name,prefix_name="_base",max_workers=1,plus_n = False,config_singleton=""):
    random_cap = lambda pax :random.randint(4,15) 
    random_slice = lambda pax,caping : random.randint(0,pax-math.ceil(pax/caping))
    solve(df, random_cap, random_slice, num_vehicles, soft_limit, file_name,prefix_name="_base_full_random",max_workers=max_workers,config_singleton=config_singleton)    
    if plus_n:
         calculating_plus_n_pax(df, caping, sclicing, num_vehicles, soft_limit, file_name,max_workers=max_workers)
        
def select_caping_slicing(df,max_cap = 15,min_cap = 4):

    caping = random.randint(min_cap,max_cap)
    df_map = list(map(lambda X : X>=min_cap,list(df["DEMAND"])))
    max_slice_fn = min(df["DEMAND"][df_map])-math.ceil(min(df["DEMAND"][df_map])/caping)
    min_slice_fn = math.ceil(min(df.DEMAND)/caping)
    slice = random.randint(min_slice_fn,max_slice_fn)
    return caping,slice

def selected_caping_slicing(df,caping,sclicing, num_vehicles, soft_limit, file_name,prefix_name="_base",max_workers=1,plus_n = False,config_singleton = ""):
    solve(df, caping, sclicing, num_vehicles, soft_limit, file_name,prefix_name="_base",max_workers=max_workers,config_singleton=config_singleton)
    if plus_n:
        calculating_plus_n_pax(df, caping, sclicing, num_vehicles, soft_limit, file_name,max_workers=max_workers)
    
    
if __name__ == '__main__':

            config_singleton = ConfigSingleton()
            verbose = True
            #caping = 10 # capping : value that force slicing eg : caping = 10; group_capacity = 12 then we slice
            #sclicing = 2 # math.ceil(n / caping)+sclicing eg : math.ceil(165/10) = 17 + 2
            max_cap = 15
            min_cap = 4
            max_workers = int(config_singleton.config["max_workers"]) # CPU number  
            soft_limit = int(config_singleton.config["soft_limit"]) # 
            if max_workers == -1:
                import os
                max_workers = os.cpu_count()
                if verbose:
                    print(f"Max workers activated : {max_workers}")
            files_name = os.listdir(f'{config_singleton.PATH}/Input')
            random.shuffle(files_name)
            for file_name in files_name:
                try:  # load json to get number of flight else set to 4:
                    with open(f'{config_singleton.PATH}/tools/all_dist_done_fr_2018_to_2020.json') as json_file:
                                    dict_older_sol = json.load(json_file)
                                    num_vehicles = dict_older_sol[file_name[:-4]]["nb_flight"]+1 
                                    # real number of vehicles not available, we only have total number of flights per day
                                    # due to high distance of DUC zone to onshore, optimized number of vehicles will not exceed "real number"
                                    # also verified empirically with ortools (cf vehicles not used)
                except:
                    num_vehicles = 4
                if num_vehicles < 4:
                    num_vehicles = 4
                df = get_dataframe(file_name,config_singleton.PATH)
                df = make_dataframe_ready(df) # convert dates in seconds

                #full_random(df,  num_vehicles, soft_limit, file_name,prefix_name="_base",max_workers=max_workers,plus_n = False,config_singleton=config_singleton)
                caping, sclicing = select_caping_slicing(df,max_cap = 15,min_cap = 4)
                print(caping, sclicing )
                selected_caping_slicing(df, caping, sclicing, num_vehicles, soft_limit, file_name,prefix_name="_base",max_workers=max_workers,plus_n = False,config_singleton=config_singleton)


        
        