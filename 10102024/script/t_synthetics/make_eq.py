# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-09-07 22:42:35
#  * @modify date 2024-09-07 22:42:35
#  * @desc [description]
#  */

import random
import pandas as pd
from GPA_01092024.tt_utils import Source,Stations
from GPA_01092024.eqscenario import EarthquakeScenario

p_dataset_path = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/tt/p_tt.h5"
s_dataset_path = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/tt/s_tt.h5"
stations_path = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/stations/standard_stations.csv"
proj = "EPSG:3857"

# stations = st

eqs = EarthquakeScenario(p_dataset_path=p_dataset_path,
                         s_dataset_path=s_dataset_path,
                         stations_path=stations_path,
                         xy_epsg=proj,
                         window=300
                         )

eqs.add_earthquakes(n_events=15,min_n_p_phases=4,
                    min_n_s_phases=2)

source = Source(latitude=31.61863,longitude=-103.99094,
                depth=4,xy_epsg=proj)
eqs.add_afterschocks(mainshock=source,n_aftershocks=15,radius_km=1,
                     min_n_p_phases=4,min_n_s_phases=2)
eqs.add_general_noise(n_phases=100)
eqs.add_noise_to_the_station(n_phases=30,n_stations=2)

eqs.remove_phases(max_hyp_distances=[10,20,30,40],
                  probabilities=[0.1,0.2,0.3,0.4],
                  min_n_p_phases=4,
                  min_n_s_phases=2
                  )

eqs.plot_phases(sort_by_source=source)
eqs.plot_phases()

output = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/eq/eqscenario.csv"
phases = eqs.get_phases()
phases.to_csv(output,index=False)
print(phases)
