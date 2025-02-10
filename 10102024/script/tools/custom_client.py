from delaware.core.client import CustomClient
from obspy import UTCDateTime


# provider = "USGS"
# client =  CustomClient(provider)
# region = [-104.84329,-103.79942,31.39610,31.91505]
# output_folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/usgs_20170101_20240922"

# years = range(2017,2025)


# for year in years:
#     print("YEAR",year)
#     try:
#         cat,picks,mag = client.get_custom_events(
#                                 minlatitude=region[2], maxlatitude=region[3], 
#                                 minlongitude=region[0], maxlongitude=region[1],
#                                 includeallorigins=True,
#                                 output_folder=output_folder,
#                                 starttime=UTCDateTime(f"{year}-01-01T00:00:00"),
#                                 endtime=UTCDateTime(f"{year+1}-01-01T00:00:00"),)
#     except Exception as e:
#         print(e)

#usgs 2020-01-02 16:03:51.348000
#texnet 2021-12-31 22:56:50.593883


#sheng
provider = "USGS"
client =  CustomClient(provider)
region = [-104.84329,-103.0733,
            31.0434,31.91505]

output_folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust_and_sheng"

years = range(2017,2025)


for year in years:
    print("YEAR",year)
    try:
        cat,picks,mag = client.get_custom_events(
                                minlatitude=region[2], maxlatitude=region[3], 
                                minlongitude=region[0], maxlongitude=region[1],
                                includeallorigins=True,
                                output_folder=output_folder,
                                starttime=UTCDateTime(f"{year}-01-01T00:00:00"),
                                endtime=UTCDateTime(f"{year+1}-01-01T00:00:00"),)
    except Exception as e:
        print(e)