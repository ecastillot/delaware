import pandas as pd
from delaware.core.event.spatial import Points


stations_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/data_git/stations/delaware_onlystations_160824.csv"
stations = pd.read_csv(stations_path)
stations["sta_id"] = 1
stations = Points(stations, xy_epsg="EPSG:3116", author=None,
                  required_columns=["latitude","longitude"],
                date_columns=["starttime"])
# print(stations.info())

