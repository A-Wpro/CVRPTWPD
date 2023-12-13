from concurrent.futures import ProcessPoolExecutor
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import copy
import math
import numpy as np
import re
import random
import pandas as pd
from tqdm import tqdm
import itertools
from itertools import product
import time
from datetime import datetime
import json
from solver import solve_vrp
from utils import test_data_consistency,get_dataframe,save_objectif
from preprocessing import main_data_model,make_dataframe_ready,generate_NN_input
from heuristic import meta_herusitic_data_model_processor
from main import *
from NN import *
from config import *

if __name__ == '__main__':

### NN block

    model = Make_NN_model()
    # Compile the model - we don't define loss here because we'll use a custom training loop
    
    config_singleton = ConfigSingleton()
    datasets,files_names = generate_NN_input(config_singleton = config_singleton)
    ### Train block
    # Training loop - replace with your actual training loop logic
    losses = []
    # Modify the training loop to train on each dataset
    for epoch in range(1000):  # Example epoch count
        total_loss = 0  # To accumulate loss over all datasets
  

        num_elements = 1#len(datasets) // 100

        # Generate a list of indices to pick
        indices = random.sample(range(len(datasets)), num_elements)

        # Pick the elements from both lists
        picked_elements_datasets = [datasets[i] for i in indices]
        picked_elements_files_names = [files_names[i] for i in indices]

        for input_data,file_name in zip(tqdm(picked_elements_datasets, total=len(picked_elements_datasets), desc="Calculating..."),picked_elements_files_names):
            loss = train_step(model, tf.expand_dims(input_data, axis=0),file_name,config_singleton)
            print(total_loss)
            total_loss += loss.numpy()
            print("**",total_loss)
        average_loss = total_loss / num_elements  # Calculate the average loss for the epoch
        losses.append(average_loss)
        print(f'Epoch {epoch}, Average Loss: {average_loss}')
    # Open the file in write mode ('w')
    with open('losses.txt', 'w') as f:
        # Write some text to the file
        f.write(str(losses))


