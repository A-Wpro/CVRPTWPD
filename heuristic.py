from concurrent.futures import ProcessPoolExecutor
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import copy
import math
import re
import pandas as pd
from tqdm import tqdm
import itertools
from itertools import product
import time
from datetime import datetime
import json
import random
from scipy.spatial import distance
import numpy as np
from solver import *
from config import *
def meta_herusitic_data_model_processor(data_models, *args):
    num_args = len(args)
    list_mode = ["brute_force", "random_brute_force",
                 "sample",   "top_less_node_first", "top_more_node_first", "top_less_node_first_random", "top_more_node_first_random"]
    if num_args == 0:
        return data_models  # brute_force
    elif num_args == 1:
        mode = args[0]
        if mode not in list_mode:
            raise f"Please select a mode in this list :{list_mode}"
        elif mode == "brute_force":
            return data_models
        elif mode == "random_brute_force":
            random.shuffle(data_models)
            return data_models
 
    elif num_args == 2:
        mode, chunk_percent = args[0], args[1]
        if mode not in list_mode:
            raise f"Please select a mode in this list :{list_mode}"
        if type(chunk_percent) != int:
            try:
                chunk_percent = int(chunk_percent)
            except:
                raise f"Please select a mode in this list :{list_mode}"
        if chunk_percent > 100:
            chunk_percent = 100
        if chunk_percent <= 0:
            chunk_percent = 1
        chunk = int(len(data_models)*(chunk_percent/100))+1
        if mode == "brute_force":

            return data_models[:chunk]
        elif mode == "random_brute_force":
            random.shuffle(data_models)
            return data_models[:chunk]
        elif mode == "sample":
            step = int(len(data_models)/(len(data_models)*(chunk_percent/100)))
            return data_models[::step]
        elif mode == "top_less_node_first":
            return sorted(data_models, key=lambda d: len(d['demands']))[:chunk]
        elif mode == "top_more_node_first":
            return sorted(data_models, key=lambda d: len(d['demands']), reverse=True)[:chunk]
        elif mode == "top_less_node_first_random":
            data_models_tmp = sorted(
                data_models, key=lambda d: len(d['demands']))[:chunk]
            random.shuffle(data_models_tmp)
            return data_models_tmp
        elif mode == "top_more_node_first_random":
            data_models_tmp = sorted(data_models, key=lambda d: len(
                d['demands']), reverse=True)[:chunk]
            random.shuffle(data_models_tmp)
            return data_models_tmp


# Compute the similarity between two input lists.
def similarity(input1, input2):
    return distance.euclidean(input1, input2)

# Heuristic algorithm to decide what inputs to test next.
def heuristic_1(solver, num_iterations=100):
    # Keeping a history of the inputs and outputs to adapt our heuristic.
    history = []
    
    # Keeping track of the best solution found so far.
    best_solution = float('inf')
    
    # Worst solutions history
    worst_solutions = []

    # Store all the similarity measures
    similarity_measures = []

    for i in range(num_iterations):
        # Generate random input to start with.
        current_input = [random.randint(1, 10) for _ in range(8)]

        # Use solver to get the output for the current input.
        current_output = solver(current_input)
        
        # Update the history
        history.append((current_input, current_output))

        # Update best solution if needed
        if current_output < best_solution:
            best_solution = current_output

        # Update worst solutions if needed
        if len(worst_solutions) < 5:  # Store up to 5 worst solutions
            worst_solutions.append((current_input, current_output))
        else:
            worst_solutions = sorted(worst_solutions, key=lambda x: x[1], reverse=True)
            if current_output < worst_solutions[0][1]:
                worst_solutions[0] = (current_input, current_output)

        # Heuristic logic to decide what to try next.
        average_output = sum(output for _, output in history) / len(history)
        
        if current_output < average_output:
            # Generate next input similar to the current best one.
            current_input = [x + random.randint(-2, 2) for x in current_input]
        else:
            # Calculate the dynamic similarity threshold based on top 10% of the iterations
            dynamic_threshold = np.percentile(similarity_measures, 90) if i > 10 else 10

            # Before generating a completely new random input, check similarity with worst cases
            while True:
                is_similar_to_worst = False
                for worst_input, _ in worst_solutions:
                    sim = similarity(current_input, worst_input)
                    similarity_measures.append(sim)

                    if sim < dynamic_threshold:
                        is_similar_to_worst = True
                        break

                if not is_similar_to_worst:
                    break
                
                # If it is similar to one of the worst solutions, regenerate the input
                current_input = [random.randint(1, 10) for _ in range(8)]

        print(f"Iteration {i+1}, Best solution so far: {best_solution}")


