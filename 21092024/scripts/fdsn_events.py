from obspy import UTCDateTime
from obspy.clients.fdsn import Client 

provider = "USGS"
client =  Client(provider)
region = [-104.84329,-103.79942,31.39610,31.91505]
try:
    cat = client.get_events(starttime=UTCDateTime("2024-04-18T23:00:00"),
                           endtime=UTCDateTime("2024-04-19T23:00:00"),
                           minlatitude=region[2], maxlatitude=region[3], 
                           minlongitude=region[0], maxlongitude=region[1])
    print(cat)
except Exception as e:
    print(e)
    print("No stream")
    
for events in cat:
    print(events)
    