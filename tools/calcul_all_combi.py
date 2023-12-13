import pandas as pd
import math
import multiprocessing as mp

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
        findCombinationsUtil(arr, index + 1, num, reducedNum - k, num_combi, cap, combinations)

def findCombinations(n, num_combi, cap):
    arr = [0] * n
    combinations = []
    findCombinationsUtil(arr, 0, n, n, num_combi, cap, combinations)
    return combinations

def calculate(cap, output_file,nc_add):
    df_prepcalc = pd.DataFrame(columns=["cap", "value", "combi","nc"])
    
    for n in range(cap, 100):

        nc = math.ceil(n / cap)+nc_add
        print(f"Processing cap: {cap}, value: {n}, number of combi: {nc}")  # print progress
        tmp = findCombinations(n, nc, cap)
        tmp_df = pd.DataFrame(columns=["cap", "value", "combi","nc"], data=[[cap, n, tmp,nc]])
        df_prepcalc = pd.concat([df_prepcalc, tmp_df])

    # Append to Parquet file after calculating all combinations for current cap
    df_prepcalc.to_parquet(output_file)
    return df_prepcalc

if __name__ == "__main__":
    output_file = fr"{config_singleton.PATH}/tools/precalculated_combinaison_heavy.parquet"
    df_prepcalc = pd.DataFrame()
    for cap in range(6, 15):
        for nc_add in range(0, 15):
            results = calculate(cap, output_file,nc_add)

    # No need to write to Parquet again here, as data is already written in calculate function
