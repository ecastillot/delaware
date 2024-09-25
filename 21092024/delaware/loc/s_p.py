# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-09-24 16:49:27
#  * @modify date 2024-09-24 16:49:27
#  * @desc [description]
#  */

import pandas as pd
from delaware.core.database import save_dataframe_to_sqlite,load_dataframe_from_sqlite
from delaware.eqviewer.eqviewer import Catalog,Picks


def get_picks(event_ids,picks_path):
    
    return load_dataframe_from_sqlite(db_name=picks_path,
                                      tables=event_ids)

def get_events(origin, picks_path, event_ids=None,
               starttime=None,endtime=None,
               region=None, agencies=None,
               region_from_src = None,
               ):
    

    origin = origin.rename(columns = {"mag":"magnitude"})
    catalog = Catalog(origin)
    
    catalog.filter("origin_time",starttime,endtime)
    
    if (region is not None) and (len(catalog) !=0):
        catalog.filter_region(region)
        
    if (region_from_src is not None) and (len(catalog) !=0):
        lat,lon, r_max, az_max =  region_from_src
        catalog.filter_by_r_az(latitude=lat,
                           longitude=lon,
                           r=r_max,
                           az=az_max)
        
    if (event_ids is not None) and (len(catalog) !=0):
        catalog.select_data({"id":event_ids})
        
    if (agencies is not None) and (len(catalog) !=0):
        catalog.select_data({"agency":agencies}) #agencies is a list
        
    if len(catalog) != 0:
        event_ids = catalog.data["id"].to_list()
        
        picks = get_picks(event_ids=event_ids,picks_path=picks_path)
        
    
    else :
        picks = pd.DataFrame(columns=["ev_id"])
    
    picks = Picks(picks)
    
    
    return catalog, picks
    