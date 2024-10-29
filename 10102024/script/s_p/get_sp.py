import pandas as pd
import datetime as dt
from delaware.utils import get_texnet_high_resolution_catalog
from delaware.core.eqviewer import Stations

picks_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust/picks.db" 
# picks_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/output/delaware_20170101_20240922/picks.db" 
high_res = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/growclust/texnet_hirescatalog.csv"

stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/stations/delaware_onlystations_160824.csv"
proj = "EPSG:3857"

## Growclust station events S-P
s_p_output = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/loc/s_p/growclust"
catalog = get_texnet_high_resolution_catalog(high_res,xy_epsg=proj)
stations = pd.read_csv(stations_path)
stations["station_index"] = stations.index
stations = Stations(stations,xy_epsg=proj)
print(catalog)
print(stations)
stations.select_data({"network":["TX","4O"]})


events = stations.get_events_by_sp(catalog,rmax=0.5,zmin=5,
                                   picks_path=picks_path,
                                   output_folder=s_p_output)
print(events)

