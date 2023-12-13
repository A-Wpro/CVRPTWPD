import numpy as np
class Pax:
    def __init__(self, id_Pax, Route_Step_by_Step, Contraints_Time,Time,Contraints_Start,Contraints_End):
        self.id_Pax = id_Pax
        self.Route_Step_by_Step =Route_Step_by_Step
        self.Contraints_Time = Contraints_Time
        self.Time = Time
        self.Contraints_Start = Contraints_Start
        self.Contraints_End = Contraints_End
        
class Route:
    def __init__(self, Path, Path_unique,Pax_By_Step):
        self.Path = Path
        self.Path_unique = Path_unique
        
class Helico:
    def __init__(self, Id_Helico, Route, Pax_List, Time,Load):
        self.Id_Helico = Id_Helico
        self.Route = Route
        self.Pax_List = Pax_List
        self.Time = Time
        self.Load = Load

class Solution:
    def __init__(self, Capacities, GPS, Name_Demands,Demands_Index,Pickups_Deliveries,Time_Windows,Demands,Distance_Matrix,Time_Matrix,Solution):
        self.Capacities = Capacities
        self.GPS = GPS
        self.Name_Demands = Name_Demands
        self.Demands_Index = Demands_Index 
        self.Pickups_Deliveries = Pickups_Deliveries 
        self.Time_Windows = Time_Windows 
        self.Demands = Demands 
        self.Distance_Matrix = Distance_Matrix 
        self.Time_Matrix = Time_Matrix 
        self.Solution = Solution 
        
    def create_fleet(self):
        
        Route_List = []
        Route_Number = []
        for k in self.Solution.keys():
            if type(k) == int:
                Route_Number.append(k)

        for nb in Route_Number:
            Route_Name = []
            for n in range(len(self.Solution[nb]["route"])):
                Route_Name.append(self.Name_Demands[self.Solution[nb]["route"][n]][:3])

            Unique_Route_Name = []
            for Name in Route_Name:
                if not Unique_Route_Name or Name != Unique_Route_Name[-1]:
                    Unique_Route_Name.append(Name)

            Route_List.append(Route(Path=Route_Name, Path_unique=Unique_Route_Name))
                
        self.Fleet = [Helico(x,Route_List[x],[],[],[]) for x in range(len(self.Capacities))]
        
Route_List = []
Route_Number = []
def find_index(n, list_of_lists):
    if n == 0:
        n = 1 
    for index, sublist in enumerate(list_of_lists):
        if n in sublist:
            return index
        
    return -1  # Return -1 if n is not found

for k in S.Solution.keys():
    if type(k) == int:
        Route_Number.append(k)
pax_list = []
for nb in Route_Number:
    Route_Name,Route_Demands,Route_Time_Matrix,Route_Load,Route_PD_Name,Route_PD_Num,Route_Time_Contraint,Pax_Route = [],[],[],[],[],[],[],[]
    for n in range(0,len(S.Solution[nb]["route"])):
        index = find_index(S.Solution[nb]["route"][n] , S.Pickups_Deliveries)
        Route_PD_Num.append(S.Pickups_Deliveries[index])
        Route_PD_Name.append([S.Name_Demands[S.Pickups_Deliveries[index][0]][:3],S.Name_Demands[S.Pickups_Deliveries[index][1]][:3]])
        Route_Time_Contraint.append(S.Time_Windows[index+1])
        
        Route_Demands.append((S.Demands[S.Solution[nb]["route"][n]]))
        Route_Name.append(S.Name_Demands[S.Solution[nb]["route"][n]][:3])
        Route_Time_Matrix.append((S.Solution[nb]["time"][n]))
        Route_Load.append((S.Solution[nb]["load"][n]))
        if n != 0 and n != len(S.Solution[nb]["route"])-1:
            ind = find_index(S.Solution[nb]["route"][n] , S.Pickups_Deliveries)
            end = S.Solution[nb]["route"].index(S.Pickups_Deliveries[ind][1])
            start = S.Solution[nb]["route"].index(S.Pickups_Deliveries[ind][0])
            r = [0 for _ in range(start)]+S.Solution[nb]["route"][start:end+1]+ [0 for _ in range(len(S.Solution[nb]["route"])-end-1)]
            r_ = []
            for i in r:
                if i != 0:
                    r_.append(S.Name_Demands[i][:3])
                else:
                    r_.append("")

            Pax_Route.append(r_)

    Unique_Route_Name,Unique_Route_Demands,Unique_Route_Time_Matrix,Unique_Route_Load = [],[],[],[]
    for Name,D,T,L in zip(Route_Name,Route_Demands,Route_Time_Matrix,Route_Load):
        if not Unique_Route_Name or Name != Unique_Route_Name[-1]:
            if len(Unique_Route_Name)>0:
                if Name != Unique_Route_Name[-1]:
                    Unique_Route_Load.append(L)
                   
            Unique_Route_Name.append(Name)
            Unique_Route_Demands.append(D)
            Unique_Route_Time_Matrix.append(T)

    ind = np.arange(len(Unique_Route_Name))

    cumpx = 0
    if len(Route_Name) >0:
        for n,l,t,i in  zip(Route_Name,Route_Load,Route_Time_Matrix,ind):
            print(n,l,m)
            end = "EBJ"
            if i+1 < len(Unique_Route_Name):
                end = Unique_Route_Name[i+1]

            for px in range(l):
                print("id_Pax ",f"{nb}_{px+cumpx}","|Route_Step_by_Step :",Pax_Route[i],",Contraints_Start",Route_PD_Name[i][0],"Contraints_End",Route_PD_Name[i][1],"| Contraints_Time",Route_Time_Contraint[i],"Time",t)
                pax_list.append(Pax(id_Pax=f"{nb}_{px+cumpx}",Route_Step_by_Step=Pax_Route,Contraints_Time=Route_Time_Contraint[i],Time=t,Contraints_Start=Route_PD_Name[i][0],Contraints_End=Route_PD_Name[i][0]))
            cumpx +=px+1



##----------
"""import pandas as pd
import os
import datetime

sol = eval(pd.read_csv(os.path.join(r'C:\Users\L1061703\workspace\pax_heli\Data\2023 _test_adam\rsp', "13-12-2019.csv_10_random_brute_force_output_30.csv"), sep=";")["data_model"].iloc[25])

S = Solution(sol["vehicle_capacities"], sol["GPS"], sol["name_demands"], sol["demands_index"], sol["pickups_deliveries"], sol["time_windows"], sol["demands"], sol["distance_matrix"], sol["time_matrix"], sol["solution"])
"""