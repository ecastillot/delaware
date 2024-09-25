import pandas as pd
import datetime as dt
from delaware.loc.s_p import *
from delaware.utils import *
from delaware.eqviewer.eqviewer import Station

origin_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/output/delaware_20170101_20240922/origin.csv"
high_res = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/seismicity/texnet_hirescatalog.csv"
picks_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/output/delaware_20170101_20240922/picks.db" 

stations_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/stations/delaware_onlystations_160824.csv"


## Growclust station events S-P
s_p_output = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/growclust"
catalog = get_texnet_high_resolution_catalog(high_res)
stations = pd.read_csv(stations_path)
stations = Station(stations)
stations.select_data({"network":["TX","4O"]})
events = stations.get_events_by_sp(catalog,rmax=0.5,zmin=5,
                                   picks_path=picks_path,
                                   output_folder=s_p_output)
print(events)


# # nlloc station events S-P
s_p_output = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/nlloc"
origin = pd.read_csv(origin_path,header=[1],parse_dates=["origin_time"])
origin = origin.rename(columns={"mag":"magnitude"})
catalog = Catalog(origin)
catalog.select_data({"agency":["TX","texnet"]})
# print(catalog)
stations = pd.read_csv(stations_path)
stations = Station(stations)
stations.select_data({"network":["TX","4O"]})
events = stations.get_events_by_sp(catalog,rmax=0.5,zmin=5,
                                   picks_path=picks_path,
                                   output_folder=s_p_output)
print(events)


# get growclust events
# origin = get_texnet_high_resolution_catalog(high_res)
# origin = origin.data
# catalog, picks = get_events(origin,picks_path)

# get nlloc events
# origin = pd.read_csv(origin_path,header=[1],parse_dates=["origin_time"])
# origin = origin.rename(columns={"mag":"magnitude"})
# catalog, picks = get_events(origin,picks_path,
#                             agencies=["TX","texnet"]
#                     # starttime= dt.datetime(2022,1,1),
#                     # endtime= dt.datetime(2022,1,4),
#                     #    region_from_src=[1.25,-104.58,0.5,360]
#                     )
# print(picks)