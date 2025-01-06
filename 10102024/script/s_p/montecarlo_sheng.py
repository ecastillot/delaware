import pandas as pd
import datetime as dt
from delaware.utils import get_texnet_high_resolution_catalog
from delaware.core.eqviewer import Stations
from delaware.loc.s_p import SP_Montecarlo,plot_montecarlo_depths_by_station
from delaware.core.read import EQPicks
from delaware.vel.vel import VelModel
import numpy as np
import os

sp_root = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/loc/sheng_s_p"
author = "growclust"
proj = "EPSG:3857"

z_array = np.arange(10,10+0.5,0.5)
for z_guess in z_array:
    # z_guess = 6
    print(f"\n################ Z: {z_guess} ################")
    root = f"/home/emmanuel/ecastillo/dev/delaware/10102024/data/loc/sheng_s_p/z_10.0/sheng/catalog"
    # catalog_path = os.path.join(root,"origin_sp_method.db")
    # picks_path = os.path.join(root,"picks_sp_method.db")
    # eqpicks = EQPicks(root,author=author,
    #                   xy_epsg=proj,
    #                   catalog_path=catalog_path,
    #                   picks_path=picks_path,
    #                   catalog_header_line=0
    #                   )
    spm = SP_Montecarlo(root,z_guess,author,proj)
    spm.run_montecarlo()
    
# plot_montecarlo_depths_by_station("/home/emmanuel/ecastillo/dev/delaware/10102024/data/loc/s_p/z_6.0/growclust/montecarlo/montecarlo.db")
    # spm.plot_stations_counts()
    # exit()
    # print(spm.catalog)
    # print(spm.picks)
    # print(spm.vel.p_vel)
    # print(spm.vel.s_vel)
    # spm.vel.plot()
    # exit()
    # sp_setup = SP_MontecarloSetup(eqpicks=eqpicks,vel_model=vel_model,
    #         stations=stations,
    #         output_folder=output_folder,
    #         z_guess=z_guess)
    # sp_setup.run()