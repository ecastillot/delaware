from delaware.core.client import CustomClient
from obspy import UTCDateTime


provider = "USGS"
client =  CustomClient(provider)
region = [-104.84329,-103.79942,31.39610,31.91505]
output_folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/usgs_20170101_20240922"

years = range(2017,2025)

#usgs 2020-01-02 16:03:51.348000
#texnet 2021-12-31 22:56:50.593883

# cat,picks,mag = client.get_custom_events(
#                                 minlatitude=region[2], maxlatitude=region[3], 
#                                 minlongitude=region[0], maxlongitude=region[1],
#                                 includeallorigins=True,
#                                 output_folder=output_folder,
#                                 starttime=UTCDateTime(f"2021-02-18 00:03:51.348000"),
#                                 endtime=UTCDateTime(f"2021-02-18 23:56:50.593883"),)

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

# cat,picks,mag = client.get_custom_events(
#                         # starttime=UTCDateTime("2024-08-02T00:00:00"),
#                         # endtime=UTCDateTime("2024-08-03T00:00:00"),
#                         minlatitude=region[2], maxlatitude=region[3], 
#                         minlongitude=region[0], maxlongitude=region[1],
#                         includeallorigins=True,
#                         output_folder=output_folder
                        
#                         # 2024
#                         starttime=UTCDateTime("2024-01-01T00:00:00"),
#                         endtime=UTCDateTime("2024-09-01T00:00:00"),
                        
#                         # # 2023
#                         # starttime=UTCDateTime("2023-01-01T00:00:00"),
#                         # endtime=UTCDateTime("2024-01-01T00:00:00"),
                        
#                         # # 2022
#                         # starttime=UTCDateTime("2022-01-01T00:00:00"),
#                         # endtime=UTCDateTime("2023-01-01T00:00:00"),
                        
#                         # # 2021
#                         # starttime=UTCDateTime("2021-01-01T00:00:00"),
#                         # endtime=UTCDateTime("2022-01-01T00:00:00"),
                        
#                         # # 2020
#                         # starttime=UTCDateTime("2020-01-01T00:00:00"),
#                         # endtime=UTCDateTime("2021-01-01T00:00:00"),
                        
#                         # # 2019
#                         # starttime=UTCDateTime("2019-01-01T00:00:00"),
#                         # endtime=UTCDateTime("2020-01-01T00:00:00"),
                        
#                         # # 2018
#                         # starttime=UTCDateTime("2018-01-01T00:00:00"),
#                         # endtime=UTCDateTime("2019-01-01T00:00:00"),
                        
#                         # # 2017
#                         # starttime=UTCDateTime("2017-01-01T00:00:00"),
#                         # endtime=UTCDateTime("2018-01-01T00:00:00"),
                        
                        
#                         )
# # cat.to_csv(cat_path,index=False)
# # print(cat)