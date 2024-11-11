import pandas as pd
import datetime as dt
from delaware.utils import get_texnet_high_resolution_catalog
from delaware.core.eqviewer import Stations
from delaware.loc.s_p import SP_MontecarloSetup,get_std_deviations
from delaware.core.read import EQPicks
from delaware.vel.vel import VelModel
import numpy as np

# picks_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust/picks.db" 
# # picks_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/output/delaware_20170101_20240922/picks.db" 
# # high_res = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/growclust/texnet_hirescatalog.csv"
# high_res = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust/origin.csv"

# stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/stations/delaware_onlystations_160824.csv"
# proj = "EPSG:3857"
# s_p_output = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/loc/s_p/growclust"

# ## Growclust station events S-P
# # catalog = get_texnet_high_resolution_catalog(high_res,xy_epsg=proj)
# stations = pd.read_csv(stations_path)
# stations["station_index"] = stations.index
# stations = Stations(stations,xy_epsg=proj)
# # print(catalog)
# print(stations)
# stations.select_data({"network":["TX","4O"]})

root = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi"
vel_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/vel/DB_model.csv"
vel_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/vel/DW_model.csv"
stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/stations/delaware_onlystations_160824.csv"
author = "growclust"
proj = "EPSG:3857"


# eqpicks = EQPicks(root=root,
#                   author=author,
#                   xy_epsg=proj,
#                   catalog_header_line=0)

# stations = pd.read_csv(stations_path)
# stations["station_index"] = stations.index
# stations = Stations(data=stations,xy_epsg=proj)
# stations.select_data({"network":["TX","4O"]})

data = pd.read_csv(vel_path)
vel_model = VelModel(data=data,dtm=-0.7,name="db1d")

x = vel_model.get_average_velocity("P",zmax=6)
print(x)

# z_array = np.arange(1,10+0.5,0.5).5

# for z_guess in z_array:
#     # z_guess = 6
#     print(f"\n################ Z: {z_guess} ################")
#     output_folder = f"/home/emmanuel/ecastillo/dev/delaware/10102024/data/loc/s_p/z_{z_guess}"
#     sp_setup = SP_MontecarloSetup(eqpicks=eqpicks,vel_model=vel_model,
#             stations=stations,
#             output_folder=output_folder,
#             z_guess=z_guess)
#     sp_setup.run()
    
    
# print(sp_method.folder_paths)
# sp_method.run_montecarlo()

# vel_model.plot_profile()
# print(catalog,picks)
# print(vel_model)
# print(eqpicks)
# print(eqpicks.__str__(extended=True))

# sp_database = SP_Database(catalog_path=catalog_path)

# events = stations.get_events_by_sp(catalog,rmax=0.5,zmin=5,
#                                    picks_path=picks_path,
#                                    output_folder=s_p_output)
# print(events)

