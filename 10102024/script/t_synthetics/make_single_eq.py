# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-09-07 22:42:35
#  * @modify date 2024-09-07 22:42:35
#  * @desc [description]
#  */

import random
import pandas as pd
import datetime as dt
from GPA_01092024.tt_utils import Source,Stations,Earthquakes
from GPA_01092024.tt import EarthquakeTravelTime
from GPA_01092024.eqscenario import EarthquakeScenario

p_dataset_path = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/tt/p_tt.npz"
s_dataset_path = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/tt/s_tt.npz"
stations_path = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/stations/standard_stations.csv"
p_single_eq_path = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/p_single_eq_2.5.csv"
s_single_eq_path = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/s_single_eq_2.5.csv"
single_eq_path = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/single_eq_2.5.csv"
proj = "EPSG:3857"

df = pd.read_csv(stations_path)
stations = Stations(data=df, xy_epsg=proj)

df = pd.DataFrame.from_dict({"latitude":[31.636749999999999],
                             "longitude":[-103.998790000000000],
                            #  "depth":[6.903000000000000],
                             "depth":[2.5000000000000],
                            #  "origin_time":[dt.datetime(2022,11,16,21,32,48.4819999)]
                             "origin_time":["2022-11-16 21:32:48.4819999"]
                             })
df["origin_time"] = pd.to_datetime(df['origin_time'])
print(df)
earthquakes = Earthquakes(data=df, xy_epsg=proj)
print(earthquakes)
print(stations)
## Loading the velocity models, and then computing the traveltimes
eq_p = EarthquakeTravelTime(phase="P", stations=stations,
                            earthquakes=earthquakes)
eq_p.load_velocity_model(path=p_dataset_path,
                            xy_epsg=proj)
p_tt = eq_p.get_traveltimes(merge_stations=True,
                            output=p_single_eq_path)
p_tt.data.sort_values(by="event_index", inplace=True)
p_tt.data["phasehint"] = "P"
print(p_tt)

eq_s = EarthquakeTravelTime(phase="S", stations=stations,
                            earthquakes=earthquakes)
eq_s.load_velocity_model(path=s_dataset_path,
                            xy_epsg=proj)
s_tt = eq_s.get_traveltimes(merge_stations=True,
                            output=s_single_eq_path)
s_tt.data.sort_values(by="event_index", inplace=True)
s_tt.data["phasehint"] = "S"

all_tt = pd.concat([p_tt.data,s_tt.data],ignore_index=True)

print(all_tt)
all_tt.to_csv(single_eq_path,index=False)
