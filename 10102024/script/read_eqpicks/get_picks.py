from delaware.core.read import EQPicks
from delaware.core.tracer import Tracer
from delaware.core.eqviewer import MulPicks,Stations,Picks
import datetime as dt
import pandas as pd

root = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi"
author = "usgs_20170101_20240922"
author2 = "pykonal_growclust"
proj = "EPSG:3857"
starttime = "2023-01-01 00:00:00"
endtime = "2023-01-03 00:00:00"

starttime = dt.datetime.strptime(starttime,
                                 "%Y-%m-%d %H:%M:%S")
endtime = dt.datetime.strptime(endtime,
                                 "%Y-%m-%d %H:%M:%S")


eqpicks = EQPicks(root=root,
                  author=author,
                  xy_epsg=proj,
                  catalog_header_line=0)
print(eqpicks.catalog)
catalog, picks = eqpicks.get_catalog_with_picks(
                                starttime=starttime,
                               endtime=endtime,
                            #    ev_ids=["texnet2023bjyx"]
                               )

eqpicks2 = EQPicks(root=root,
                  author=author2,
                  xy_epsg=proj,
                  catalog_header_line=0)
print(eqpicks2.catalog)
catalog2, picks2 = eqpicks2.get_catalog_with_picks(
                                starttime=starttime,
                               endtime=endtime,
                               )
picks2.p_color = "cyan"
picks2.s_color = "magenta"

mulpicks = MulPicks([picks,picks2])
print(mulpicks)

stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/stations/delaware_onlystations_160824.csv"
wav_starttime = "2023-01-01 15:26:30"
wav_endtime = "2023-01-01 15:27:00"

stations = pd.read_csv(stations_path)
stations["station_index"] = stations.index
stations = Stations(data=stations,xy_epsg=proj)

wav_starttime = dt.datetime.strptime(wav_starttime,
                                 "%Y-%m-%d %H:%M:%S")
wav_endtime = dt.datetime.strptime(wav_endtime,
                                 "%Y-%m-%d %H:%M:%S")

tracer = Tracer(url="texnet",mulpicks=mulpicks,stations=stations,
                preferred_author=author)
tracer.plot(wav_starttime,wav_endtime,network_list=["TX"],
            remove_stations=["PCOS","VHRN","ALPN"],
            sort_by_first_arrival=True)
# dw.get_waveforms(network="")