from delaware.core.read import EQPicks
from delaware.core.tracer import Tracer
from delaware.core.eqviewer import MulPicks,Stations,Picks
import datetime as dt
import pandas as pd
import os

root = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi"
stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/stations/delaware_onlystations_160824.csv"
# author = "usgs_20170101_20240922"
author = "growclust"
author2 = "pykonal_growclust"
proj = "EPSG:3857"
starttime = "2023-01-01 00:00:00"
endtime = "2023-01-03 00:00:00"
x = (-104.84329,-103.79942)
y = (31.3961,31.91505)
z = (-2,20)
    
    
starttime = dt.datetime.strptime(starttime,
                                 "%Y-%m-%d %H:%M:%S")
endtime = dt.datetime.strptime(endtime,
                                 "%Y-%m-%d %H:%M:%S")

stations = pd.read_csv(stations_path)
stations["station_index"] = stations.index
stations = Stations(data=stations,xy_epsg=proj)

eqpicks = EQPicks(root=root,
                  author=author,
                  xy_epsg=proj,
                  catalog_header_line=0)
print(eqpicks.catalog)
catalog, picks = eqpicks.get_catalog_with_picks(
                                starttime=starttime,
                               endtime=endtime,
                               region_lims= list(x+y),
                               stations=stations
                            #    ev_ids=["texnet2023bjyx"]
                               )

picks.author = "texnet"

eqpicks2 = EQPicks(root=root,
                  author=author2,
                  xy_epsg=proj,
                  catalog_header_line=0)
print(eqpicks2.catalog)
catalog2, picks2 = eqpicks2.get_catalog_with_picks(
                                starttime=starttime,
                               endtime=endtime,
                               region_lims= list(x+y),
                               stations=stations
                               )
picks2.p_color = "cyan"
picks2.s_color = "magenta"
picks2.author = "pykonal"

# mulpicks = MulPicks([picks,picks2])
mulpicks = MulPicks([picks,picks2])
x = mulpicks.compare("texnet","pykonal")
# print(x[["sr_r [km]_texnet","sr_r [km]_pykonal"]])
# print(x)
# print(mulpicks)

# stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/stations/delaware_onlystations_160824.csv"
# wav_starttime = "2023-01-01 15:26:33.00"
# wav_endtime = "2023-01-01 15:26:36.40"
# # wav_starttime = "2023-01-01 15:26:30"
# # wav_endtime = "2023-01-01 15:27:00"

# stations = pd.read_csv(stations_path)
# stations["station_index"] = stations.index
# stations = Stations(data=stations,xy_epsg=proj)

# wav_starttime = dt.datetime.strptime(wav_starttime,
#                                  "%Y-%m-%d %H:%M:%S.%f")
# wav_endtime = dt.datetime.strptime(wav_endtime,
#                                  "%Y-%m-%d %H:%M:%S.%f")

# trace_output_folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq_seismograms"
# fig_output_folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/figures"

# out_fmt = "%Y%m%dT%H%M%Sz%f"
# name = f"wav_{wav_starttime.strftime(out_fmt)}_{wav_endtime.strftime(out_fmt)}"
# trace_name = f"{name}.mseed"
# fig_name = f"{name}.png"


# trace_output = os.path.join(trace_output_folder,trace_name)
# fig_output = os.path.join(fig_output_folder,fig_name)
# tracer = Tracer(url="texnet",mulpicks=mulpicks,stations=stations,
#                 preferred_author=picks.author )
# tracer.plot(wav_starttime,wav_endtime,
#             network_list=["TX"],
#             stations_list=["PB38","PB37"],
#             # stations_list=["PB35","PB36","PB28","PB38","PB37"],
#             remove_stations=["PCOS","VHRN","ALPN"],
#             # trace_output=trace_output,
#             sort_by_first_arrival=True,
#             savefig=fig_output)
# # dw.get_waveforms(network="")