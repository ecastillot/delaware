from obspy import UTCDateTime
from delaware.core.client import CustomClient
import pandas as pd

provider = "IRIS"
client =  CustomClient(provider)
# region = [-104.84329,-103.79942,31.39610,31.91505]
region = [-103.973638,-103.963891,31.607104,31.614540]

cat,picks,mag = client.get_custom_events(
                        minlatitude=region[2], maxlatitude=region[3], 
                        minlongitude=region[0], maxlongitude=region[1],
                        includeallorigins=True,
                        starttime=UTCDateTime(f"2024-01-01 15:26:30"),
                        endtime=UTCDateTime("2024-01-30 15:26:32"))