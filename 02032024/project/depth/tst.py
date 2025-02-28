#               ev_id station         r          az      tt_P      tt_S     ts-tp
# 0     texnet2023vmel    PB36  2.839433  283.493741 -1.215999  3.154001  4.370000

from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client

client = Client("USGS")
catalog = client.get_events( starttime=UTCDateTime("2023-12-20 04:57:36.564"), 
                            endtime=UTCDateTime("2023-12-20 04:57:56.564"),
                            # includeallarrivals=True,
                            eventid="tx2023yvik",
                            # catalog="texnet"
                            )
print(catalog[0].picks)

# texnet2023yvik 