import pandas as pd
from delaware.core.event.stations import Stations
from delaware.core.event.events import get_texnet_high_resolution_catalog

output_folder = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/data"
events_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust/origin_r.csv"
picks_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust/picks.db"
stations_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/data_git/stations/delaware_onlystations_160824.csv"
proj = "EPSG:3857"
author= "texnet"
x = (-104.84329,-103.79942)
y = (31.3961,31.91505)

stations = pd.read_csv(stations_path)
sta_id = lambda x: ".".join((x.network,x.station))
stations["sta_id"] = stations.apply(sta_id, axis=1)
stations = Stations(stations, xy_epsg=proj , author="texnet")
stations.filter_rectangular_region(x+y)


events = get_texnet_high_resolution_catalog(events_path,xy_epsg=proj,
                                            author=author)
print(events)
sp_events = stations.get_events_by_sp(catalog=events,rmax=1,
                          picks_path=picks_path,
                          output_folder=output_folder)
print(sp_events)


