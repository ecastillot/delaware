import random
import pandas as pd
from delaware.synthetic.tt import EarthquakeTravelTimeDataset
from delaware.utils import get_db_stations


stations_path = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/stations/delaware_onlystations_160824.csv"
tt_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/synthetics/tt/P_tt.h5"


proj = "EPSG:3857"

x = (-104.84329,-103.79942)
y = (31.3961,31.91505)
z = (-2,20)
nx, ny, nz = 100,60,100


# preparing stations data
stations = get_db_stations(stations_path,x,y,proj)
ptt = EarthquakeTravelTimeDataset("P",stations)

while True:
    n_event = int(random.uniform(0,nx*ny*nz))
    tt = ptt.read_traveltimes_from_single_earthquake(tt_path,
                                        event_id=n_event,
                                        merge_stations=True)
    print(tt)
    tt.plot(stations)