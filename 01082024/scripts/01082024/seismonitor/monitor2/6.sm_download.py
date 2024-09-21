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

monitor_path = "/home/emmanuel/ecastillo/dev/delaware/seismonitor/monitor2"

sgc_rest = WaveformRestrictions(network="TX,4O",
                    station="*",
                    location="*",
                    channel="*",
                    starttime=UTCDateTime("2023-12-07T04:03:00.000000Z"),
                    endtime=UTCDateTime("2023-12-07T04:08:00.000000Z"),
                    location_preferences=["","00","20","10","40"],
                    channel_preferences=["HH","CH","BH","EH","HN","HL"],
                    filter_networks=[], 
                    filter_stations=[],
                    filter_domain= [-104.84329,-103.79942,31.3961,31.91505],
                    # filter_domain= [-103.6146,-102.9448,31.0092 ,31.2747],
                    )
sgc_client = FDSNClient('TEXNET')


sgc_provider = Provider(sgc_client,sgc_rest)
md = MseedDownloader(providers=[sgc_provider])

stations_path = os.path.join(monitor_path,"stations")
downloads_path = os.path.join(monitor_path,"downloads")
mseed_storage = os.path.join(downloads_path,"{station}/{network}.{station}.{location}.{channel}__{starttime}__{endtime}.mseed")

inv,json = md.make_inv_and_json(stations_path)
md.download(mseed_storage,
            picker_args={"batch_size":1,
                        "overlap":0.3,"length":60},
            chunklength_in_sec=300,n_processor=None)