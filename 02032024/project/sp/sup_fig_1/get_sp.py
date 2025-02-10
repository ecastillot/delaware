import pandas as pd
import os
from delaware.core.event.stations import Stations
from delaware.core.event.events import get_texnet_high_resolution_catalog,get_texnet_original_usgs_catalog

events_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust_and_sheng/origin.csv"
picks_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/usgs_20170101_20240922/picks.db"
output_folder = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/data/sup_fig_1"
stations_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/data_git/stations/delaware_onlystations_160824.csv"
proj = "EPSG:3857"
author= "texnet"

only_stations = ["PB37","PB28","PB35","PB36","SA02","PB24","WB03"]
sheng_stations = ["PB04","PB16"]
x = (-104.84329,-103.79942)
y = (31.3961,31.91505)
# radii = [1,2,3,4,5]
radii = [10]

stations_df = pd.read_csv(stations_path)
sta_id = lambda x: ".".join((x.network,x.station))
stations_df["sta_id"] = stations_df.apply(sta_id, axis=1)

sheng_stations_df = stations_df[stations_df["station"].isin(sheng_stations)]


stations = Stations(stations_df, xy_epsg=proj , author="texnet")


stations.filter_rectangular_region(x+y)
stations.append(sheng_stations_df)

sta_data = stations.data.drop_duplicates("station",ignore_index=True)
        
all_events = dict((sta,[]) for sta in sta_data["station"].to_list())
all_picks = dict((sta,[]) for sta in sta_data["station"].to_list())
events = get_texnet_high_resolution_catalog(events_path,xy_epsg=proj,
                                            author=author)

for r in radii:
    sta = stations.copy()
    out = os.path.join(output_folder,f"{r}_km")
    # print(list(set(sta["station"].to_list())))
    sp_events = sta.get_events_by_sp(catalog=events,rmax=r,
                            picks_path=picks_path,
                            output_folder=out)

    print(sp_events)


