import pandas as pd
from delaware.core.event.data import DataFrameHelper


stations_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/data_git/stations/delaware_onlystations_160824.csv"
stations = pd.read_csv(stations_path)
stations = DataFrameHelper(stations,
                           required_columns=["latitude","longitude"],
                          date_columns=["starttime"])
print(stations)

