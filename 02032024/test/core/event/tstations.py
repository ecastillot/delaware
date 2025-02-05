import pandas as pd
from delaware.core.event.stations import Stations


stations_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/data_git/stations/delaware_onlystations_160824.csv"
proj = "EPSG:3857"
x = (-104.84329,-103.79942)
y = (31.3961,31.91505)

stations = pd.read_csv(stations_path)
sta_id = lambda x: ".".join((x.network,x.station))
stations["sta_id"] = stations.apply(sta_id, axis=1)
stations = Stations(stations, xy_epsg=proj , author="texnet")

stations.filter_rectangular_region(x+y)

data = stations.data
data.to_csv("/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview/aoi_stations.csv")
print(stations)


