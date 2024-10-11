import random
import pandas as pd
from GPA_01092024.tt_utils import Stations,single_latlon2xy_in_km
from GPA_01092024.tt import EarthquakeTravelTimeDataset

stations_path = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/stations/delaware_onlystations_160824.csv"
tt_path = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/tt/s_tt.h5"

proj = "EPSG:3857"

x = (-104.84329,-103.79942)
y = (31.3961,31.91505)
z = (-2,10)
nx, ny, nz = 60,60,60


# preparing stations data
stations_data = pd.read_csv(stations_path)
stations_data.rename(columns={
                            "station_elevation":"elevation"},
                            )
stations_data["station_index"] = stations_data.index
stations_data["elevation"] = stations_data["elevation"]/1e3

#stations inside the polygon
dw_w_pol = [(x[1],y[0]),
        (x[1],y[1]),
        (x[0],y[1]),
        (x[0],y[0]),
        (x[1],y[0])]
stations = Stations(stations_data,xy_epsg=proj)
stations.filter_region(polygon=dw_w_pol)


ptt = EarthquakeTravelTimeDataset("P",stations)

while True:
    n_event = int(random.uniform(0,nx*ny*nz))
    # n_event = 180906
    print(n_event)
    tt = ptt.read_traveltimes_from_single_earthquake(tt_path,
                                        event_id=n_event,
                                        merge_stations=True)
    print(tt)
# # print(tt.data)
    tt.plot(stations)