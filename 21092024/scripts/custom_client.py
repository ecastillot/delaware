from delaware.core.client import CustomClient
from obspy import UTCDateTime


provider = "USGS"
client =  CustomClient(provider)
region = [-104.84329,-103.79942,31.39610,31.91505]
output_folder = "/home/emmanuel/ecastillo/dev/delaware/21092024/output/delaware"
cat,picks,mag = client.get_custom_events(starttime=UTCDateTime("2024-04-18T23:00:00"),
                        endtime=UTCDateTime("2024-04-19T23:00:00"),
                        minlatitude=region[2], maxlatitude=region[3], 
                        minlongitude=region[0], maxlongitude=region[1],
                        includeallorigins=True,
                        output_folder=output_folder
                        )
# cat.to_csv(cat_path,index=False)
print(cat)