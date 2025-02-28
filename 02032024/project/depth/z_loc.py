import pandas as pd
import os
import string
import numpy as np
import matplotlib.pyplot as plt
from delaware.vel.vel import VelModel
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

mydata = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/wells_aoi_all.csv"
formations = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/env_csv-FormationTops-332ba_2024-12-23.csv"
stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/delaware_onlystations_160824.csv"
output_fig = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/depth/z.png"
sp_path ="/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/fig_1/fig_1_picks.csv"

custom_palette = {"PB36": {"color":"#26fafa","region":1,"vp/vs":1.72}, 
                  "PB35": {"color":"#2dfa26","region":1,"vp/vs":1.69}, 
                  "PB28": {"color":"#ad16db","region":1,"vp/vs":1.70}, 
                  "PB37": {"color":"#1a3be3","region":1,"vp/vs":1.70}, 
                  "SA02": {"color":"#f1840f","region":2,"vp/vs":1.72}, 
                  "WB03": {"color":"gray","region":3,"vp/vs":1.65}, 
                  "PB24": {"color":"#0ea024","region":3,"vp/vs":1.65}, 
                  }
vp_regions = {1:"/home/emmanuel/ecastillo/dev/delaware/02032024/project/vp/R1.csv",
              2:"/home/emmanuel/ecastillo/dev/delaware/02032024/project/vp/R2.csv",
              3:"/home/emmanuel/ecastillo/dev/delaware/02032024/project/vp/R3.csv"}
# vp_regions = { i: pd.read_csv(vp_regions[i]) for i in vp_regions.keys()}

for r,vp in vp_regions.items():
    data = pd.read_csv(vp)
    data = data.rename(columns={"Depth[km]":"Depth (km)","Vp_mean[km/s]":"VP (km/s)"})
    data["VS (km/s)"] = data["VP (km/s)"] / 1.732
    vel = VelModel(data,name=f"R{r}")
    # z_values = np.linspace(-1.2,12,100)
    # avg_vel = [vel.get_average_velocity(phase_hint="P", zmax=z) \
    #                                                 for z in z_values]
    # avg_vel = np.array(avg_vel)
    vp_regions[r] = vel
#     print(avg_vel[1::],z_values[1::])
#     exit()
# def get_average_vel()
#     avg_vel = [my_vel.get_average_velocity(phase_hint="P", zmax=z) \
#                                                     for z in z_values]

# custom_palette = {"PB36": {"color":"#26fafa","region":1,"vp/vs":1.725}, 
#                   "PB35": {"color":"#2dfa26","region":1,"vp/vs":1.725}, 
#                   "PB28": {"color":"#ad16db","region":1,"vp/vs":1.725}, 
#                   "PB37": {"color":"#1a3be3","region":1,"vp/vs":1.725}, 
#                   "SA02": {"color":"#f1840f","region":2,"vp/vs":1.69}, 
#                   "WB03": {"color":"gray","region":3,"vp/vs":1.61}, 
#                   "PB24": {"color":"#0ea024","region":3,"vp/vs":1.61}, 
#                   }

picks = pd.read_csv(sp_path )
picks = picks[picks["station"].isin(list(custom_palette.keys()))]

data = pd.read_csv(mydata)
formations = pd.read_csv(formations)

form_rl1 = "42-109-31395-00-00"
form_r1 = "42-109-31382-00-00"
form_r2 = "42-109-30824-00-00"
form_r3 = "42-109-31383-00-00"
form_rr1 = "42-389-32460-00-00"

row_rl1 = "42-109-31395-00-00"
row_r1 = "42-109-00406-00-00"
row_r2 = "42-109-32455-00-00"
row_r3 = "42-109-31383-00-00"
row_rr1 = "42-389-32460-00-00"

well_rl1 = ["42-109-31395-00-00"]
well_r1 = ["42-109-00406-00-00"]
well_r2 = ["42-109-32255-00-00","42-109-32455-00-00"]
well_r3 = ["42-109-31383-00-00","42-109-31362-00-00",
           "42-109-31375-00-00","42-389-33816-00-00",
           "42-389-32876-00-00",
         #   "42-389-32460-00-00"
           ]
well_rr1 = ["42-389-32460-00-00"]

data_rl1 = data[data["well_name"].isin(well_rl1)]
data_rr1 = data[data["well_name"].isin(well_rr1)]
data1 = data[data["well_name"].isin(well_r1)]
data2 = data[data["well_name"].isin(well_r2)]
data3 = data[data["well_name"].isin(well_r3)]

custom_palette["PB36"]["data"] = data1
custom_palette["PB35"]["data"] = data1
custom_palette["PB28"]["data"] = data1
custom_palette["PB37"]["data"] = data1
custom_palette["SA02"]["data"] = data2
custom_palette["WB03"]["data"] = data3
custom_palette["PB24"]["data"] = data3

custom_palette["PB36"]["form"] = form_r1
custom_palette["PB35"]["form"] = form_r1
custom_palette["PB28"]["form"] = form_r1
custom_palette["PB37"]["form"] = form_r1
custom_palette["SA02"]["form"] = form_r2
custom_palette["WB03"]["form"] = form_r3
custom_palette["PB24"]["form"] = form_r3

stations = pd.read_csv(stations_path)
stations_columns = ["network","station","latitude","longitude","elevation"]
stations = stations[stations_columns]

stations = stations[stations["station"].isin(list(custom_palette.keys()))]

stations_with_picks = list(set(picks["station"].to_list()))
order = stations.copy()
order = order[order["station"].isin(stations_with_picks)]
order = order.sort_values("longitude",ignore_index=True,ascending=True)
order = order.drop_duplicates(subset="station")
order = order["station"].to_list()

# print(order)
# exit()

# cond = (formations["API_UWI_12"] == form_r1[:-3]) |\
#          (formations["API_UWI_12"] == form_r2[:-3]) |\
#          (formations["API_UWI_12"] == form_r3[:-3] )
# formations = formations[cond]

picks["vp/vs"] = picks.apply(lambda x: custom_palette[x["station"]]["vp/vs"],axis=1)
picks["region"] = picks.apply(lambda x: custom_palette[x["station"]]["region"],axis=1)
picks["coef"] = picks.apply(lambda x: x["ts-tp"]/(x["vp/vs"]-1),axis=1)

depths = np.linspace(-1.2,20,200)

station = "PB36"

reloc_events = []
for station in custom_palette.keys():

  station_picks = picks[picks["station"]==station]
  station_picks.reset_index(inplace=True,drop=True)
  station_picks["z"] = np.nan
  for depth in depths:
    avg_vp = vel.get_average_velocity(phase_hint="P", zmax=depth)
    
    if avg_vp is None:
      continue
    
    station_picks["teo_z"] = station_picks["coef"]*avg_vp
    
    station_picks["f"] = (depth - station_picks["teo_z"] > 0) 
    
    # print(station_picks["z"] is np.nan)
    calculate_z = lambda x: depth if ((x["f"] == True) & pd.isna(x["z"])) else x["z"]
    station_picks["z"] = station_picks.apply(calculate_z ,axis=1)
    
  station_picks.dropna(subset=["z"],inplace=True)
  reloc_events.append(station_picks)

reloc_events = pd.concat(reloc_events)

high_res_events = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust_and_sheng/origin.csv"
high_res_events = pd.read_csv(high_res_events)
reloc_events = pd.merge(reloc_events,high_res_events,on="ev_id")
reloc_events = reloc_events[["ev_id","origin_time","latitude","longitude","depth","z","station","ts-tp","vp/vs","region"]]
reloc_events.to_csv("/home/emmanuel/ecastillo/dev/delaware/02032024/project/depth/reloc_events.csv",
                    index=False)
print(reloc_events)
  # print(avg_vp )
  # exit()
    # teo_vel = station_picks["coef"]*