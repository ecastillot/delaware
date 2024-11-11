from obspy import UTCDateTime


provider = "USGS"
client =  CustomClient(provider)

cat,picks,mag = client.get_custom_events(
                        minlatitude=region[2], maxlatitude=region[3], 
                        minlongitude=region[0], maxlongitude=region[1],
                        includeallorigins=True,
                        starttime=UTCDateTime(f"{year}-01-01T00:00:00"),
                        endtime=UTCDateTime(f"{year+1}-01-01T00:00:00"),)