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


models = os.path.join(os.path.dirname(os.getcwd()),"picking_models")
clone_seismonitor_data(models,branch="models")
eqt_model = os.path.join(models,"EQTransformer_models","EqT_model.h5")

print("Models dir: ",models)
print("Important folders in your models",os.listdir(models))

eqtobj = EQTransformerObj(model_path=eqt_model,
            n_processor = 6,
            overlap = 0.3,
            detection_threshold =0.01,
            P_threshold = 0.01,
            S_threshold = 0.01,
            batch_size = 1,
            number_of_plots = 0,
            plot_mode = 1 ) 

out_dir = os.path.join(monitor_path ,"picks","eqt")
result = os.path.join(monitor_path ,"picks","eqt","seismonitor_picks.csv")

downloads_path = "/home/emmanuel/ecastillo/dev/delaware/seismonitor/monitor/downloads"
stations_path = "/home/emmanuel/ecastillo/dev/delaware/seismonitor/monitor/stations"

eqt = EQTransformer(eqtobj)
eqt.pick(downloads_path,stations_path,out_dir)
piut.eqt_picks_2_seismonitor_fmt(out_dir,downloads_path,result)