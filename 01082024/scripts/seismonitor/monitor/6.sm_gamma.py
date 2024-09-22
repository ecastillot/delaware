import os
import pandas as pd
from SeisMonitor.monitor.picker.ai import EQTransformer,EQTransformerObj
from SeisMonitor.monitor.picker import utils as piut
from SeisMonitor.monitor.associator.ai import GaMMA,GaMMAObj
from SeisMonitor.monitor.associator import utils as asut
from SeisMonitor.monitor.locator.nlloc.nlloc import NLLoc
from SeisMonitor.monitor.locator import utils as lut
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client as FDSNClient
from SeisMonitor.monitor.magnitude.mag import Magnitude
from SeisMonitor.core.objects import WaveformRestrictions,Provider
from SeisMonitor.monitor.downloader.seismonitor import MseedDownloader

from SeisMonitor.utils4examples import clone_seismonitor_data

monitor_path = "/home/emmanuel/ecastillo/dev/delaware/seismonitor/monitor"
stations_path = "/home/emmanuel/ecastillo/dev/delaware/seismonitor/monitor/stations"
result = os.path.join(monitor_path ,"picks","eqt","seismonitor_picks.csv")
gamma_region = [-104.84329,-103.79942,31.3961,31.91505,0,10]
gamma_proj = "EPSG:3857"

gc = GaMMAObj(gamma_region,gamma_proj,
                use_amplitude = False,
                use_dbscan=False,
                calculate_amp=False)

inv = os.path.join(stations_path,"inv.xml")
gamma_dir = os.path.join(monitor_path,"gamma_asso","eqt")


g = GaMMA(gc)
obspy_catalog, df_catalog,df_picks = g.associate(result,inv,gamma_dir)
print(obspy_catalog)