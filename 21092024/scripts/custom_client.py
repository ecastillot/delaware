from delaware.core.client import CustomClient
from obspy import UTCDateTime


provider = "USGS"
client =  CustomClient(provider)
region = [-104.84329,-103.79942,31.39610,31.91505]
cat = client.get_custom_events(starttime=UTCDateTime("2024-04-18T23:00:00"),
                        endtime=UTCDateTime("2024-04-19T23:00:00"),
                        minlatitude=region[2], maxlatitude=region[3], 
                        minlongitude=region[0], maxlongitude=region[1],
                        includeallorigins=True,
                        #eventid="tx2024hstr",
                        #includeallmagnitudes=True,
                        )
print(cat)