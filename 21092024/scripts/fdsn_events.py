from obspy import UTCDateTime
from obspy.clients.fdsn import Client 

provider = "USGS"
client =  Client(provider)
region = [-104.84329,-103.79942,31.39610,31.91505]
try:
    cat = client.get_events(
                starttime=UTCDateTime("2022-11-16T21:30:00"),
                           endtime=UTCDateTime("2022-11-16T21:33:00"),
                        #    minlatitude=region[2], maxlatitude=region[3], 
                        #    minlongitude=region[0], maxlongitude=region[1]
                        includeallorigins=True,
                        eventid="tx2022wmmd"
                           )
    print(cat)
except Exception as e:
    print(e)
    print("No stream")
 
    
for events in cat:
    magnitude = events.preferred_magnitude()
    origin = events.preferred_origin()
    # print(magnitude)
    print(origin)
    
print(cat)
output = "/home/emmanuel/ecastillo/dev/delaware/21092024/output/nlloc_test_08102024/test_cat.xml"
cat.write(output,format="QUAKEML")