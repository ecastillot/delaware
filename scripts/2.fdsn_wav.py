from obspy import UTCDateTime
from obspy.clients.fdsn import Client 

provider = "TEXNET"
client =  Client(provider)
try:
    st = client.get_waveforms(
                                # network="TX",station="PB23",
                                network="4O",station="WB09",
                                location="00",channel="HH?",
                                starttime=UTCDateTime("2024-04-17 00:00:00"),
                                endtime=UTCDateTime("2024-04-18 00:00:00"))
    print(st)
except Exception as e:
    print(e)
    print("No stream")