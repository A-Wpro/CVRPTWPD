import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import MultiHeadAttention, Dropout, LayerNormalization
from keras import regularizers
import numpy as np
from solver import solve_vrp
from utils import *
from main import solve
import json
import os
from preprocessing import main_data_model,make_dataframe_ready
from config import *
'''
todo : 
    dropout
    multihead ? 
    add norm ?


'''
import tensorflow as tf

def custom_activation(x):
    return tf.where(x <= 1.0, 1.0, tf.where(x > 15.0, 15.0, x))


from keras import initializers

def Make_NN_model():
    # Define the neural network architecture
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(150,)),
        layers.Dense(64, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)),
        layers.Dense(2, activation=custom_activation)  # Output layer with 2 neurons
    ])
    return model



# Placeholder solver function - replace with your actual solver logic
def solver_(slicing, caping,file_name,config_singleton):
    slicing = tf.cast(slicing, tf.float32)
    caping = tf.cast(caping, tf.float32)
    print(slicing,caping)
    global_malus = 1000000000.0
    malus = tf.where(tf.logical_or(tf.logical_or(slicing <= 1, caping > 15), caping <= 4), global_malus*tf.abs(slicing) + global_malus*tf.abs(caping)+global_malus+1, 0.0)
    print(tf.cast(malus, tf.float32))
    if tf.cast(malus, tf.float32) < 1: 
        max_workers = -1 # CPU number 
        soft_limit = int(config_singleton.config["soft_limit"]) # 
        
        if max_workers == -1:
            import os
            max_workers = os.cpu_count()

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
        sol = solve(df, caping = caping, sclicing = slicing, num_vehicles = num_vehicles, soft_limit = soft_limit, file_name =  file_name,verbose= True,config_singleton = config_singleton,max_workers = os.cpu_count() ,save = False)

        if len(sol) == 0 or [sol[x]["solution"] for x in range(len(sol))] == ["no solution found"]*len(sol):
             return -(tf.abs(slicing)*global_malus + tf.abs(caping)*global_malus+global_malus) # Cost func
        solution_value = [global_malus if "no solution found" in sol[x]["solution"] else sol[x]["solution"]["Total distance of all routes"]  for x in range(len(sol))]
        return -(tf.abs(slicing) + tf.abs(caping) +  min(solution_value)//1000) # Cost func
    
    return -(tf.abs(slicing) * tf.abs(caping) * malus+malus*malus)

 
# RL

def train_step(model, input_data,file_name,config_singleton):
    optimizer = tf.keras.optimizers.Adam()

    with tf.GradientTape() as tape:
        # Forward pass: Compute the slicing and caping values from the neural network
        predictions = model(input_data, training=True)  # This will be a batch of predictions
        slicing = predictions[:, 0] 
        caping = predictions[:, 1]    
        rewards = solver_(slicing, caping,file_name,config_singleton)
        loss = -tf.reduce_mean(rewards)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss