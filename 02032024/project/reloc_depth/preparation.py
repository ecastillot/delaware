from project.reloc_depth.utils import latlon2yx_in_km
#               ev_id station         r          az      tt_P      tt_S     ts-tp
# 0     texnet2023vmel    PB36  2.839433  283.493741 -1.215999  3.154001  4.370000
from delaware.core.event.events import get_texnet_high_resolution_catalog
import pandas as pd
import numpy as np

events_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust/origin.csv"

relocz_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/depth/reloc_events.csv"
out = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/reloc_depth/events_initial.csv"

cross_elv_data = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/basement/cross_elevation_plot_data_31.7_-104.8_31.7_-103.8.csv")  # Update with actual file path
cross_elv_data["Elevation"] = cross_elv_data["Elevation"]*-1/1e3

proj = "EPSG:3857"
author= "texnet"

#events
events = get_texnet_high_resolution_catalog(events_path,xy_epsg=proj,
                                            author=author)


data = events.data

data["depth_from_sea_level"] = data["depth"].copy()
data["depth_from_surface"] = data["depth"] - np.interp(data["longitude"], 
                                            cross_elv_data["Longitude"], 
                                            cross_elv_data["Elevation"])
data["depth"] = data["depth_from_surface"]
data = data[["ev_id","origin_time","latitude","longitude","depth","magnitude",
             "depth_from_sea_level","depth_from_surface"]]
# print(data)
# print(data[data["ev_id"]=="texnet2021mvfp"])
# exit()

relocz_data = pd.read_csv(relocz_path,parse_dates=["origin_time"])
relocz_data["z_from_surface"] = relocz_data["z"]
relocz_data["z_from_sea_level"] = relocz_data["z"] + np.interp(relocz_data["longitude"], 
                                                   cross_elv_data["Longitude"], 
                                                   cross_elv_data["Elevation"])
relocz_data = relocz_data[["ev_id","z","station","region","z_from_surface","z_from_sea_level"]]
relocz_data = relocz_data.rename(columns={"z":"depth"})

data = pd.merge(data,relocz_data,on="ev_id",how="left",suffixes=('_TexNet_GrowClust','_S-P_ZReloc'))

# Applying the get_xy function to each row of the DataFrame
data = latlon2yx_in_km(data, proj)
data.to_csv(out,index=False)
print(data.info())

#stations
# stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/delaware_onlystations_160824.csv"
# custom_palette = {"PB35": "#26fafa", 
#                   "PB36": "#2dfa26", 
#                   "PB28": "#ad16db", 
#                   "PB37": "#1a3be3", 
#                   "WB03": "#ffffff", 
#                   "SA02": "#f1840f", 
#                   "PB24": "#0ea024", 
#                   }
# stations = pd.read_csv(stations_path)
# stations_columns = ["network","station","latitude","longitude","elevation"]
# stations = stations[stations_columns]
# stations = stations[stations["station"].isin(list(custom_palette.keys()))]
# stations = latlon2yx_in_km(stations, proj)

# out = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/reloc_depth/stations.csv"
# stations.to_csv(out,index=False)
# print(stations)