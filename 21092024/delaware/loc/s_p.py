# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-09-24 16:49:27
#  * @modify date 2024-09-24 16:49:27
#  * @desc [description]
#  */

import pandas as pd
import matplotlib.pyplot as plt
from delaware.core.database import save_dataframe_to_sqlite,load_dataframe_from_sqlite
from delaware.eqviewer.eqviewer import Catalog,Picks


class SP_Database():
    def __init__(self,catalog_path,picks_path):
        self.sp_catalog = load_dataframe_from_sqlite(db_name=catalog_path)
        self.sp_picks = load_dataframe_from_sqlite(db_name=picks_path)
        
        # print(sp_catalog)
        # print(sp_picks)
    
    @property
    def n_stations(self):
        stations = self.sp_picks.drop_duplicates("station_code")
        n_stations = len(stations)
        return n_stations
    
    @property
    def n_events(self):
        events = self.sp_picks.drop_duplicates("ev_id")
        n_events = len(events)
        return n_events
    
    def __str__(self) -> str:
        
        msg = f"Stations | {self.n_stations} stations, {self.n_events} events "
        return msg

    def plot_stations_counts(self):
        
        # First, count the number of occurrences of each station_name
        station_counts = self.sp_picks.groupby('station_code')['ev_id'].count()

        # Plot the histogram
        fig,ax = plt.subplots(1,1,figsize=(10, 6))
        # plt.figure(figsize=(10, 6))
        station_counts.plot(kind='bar',color="coral")
        
        for idx, value in enumerate(station_counts):
            ax.text(idx, value, f'{value}', ha='center', va='bottom', fontsize=14)
        
        ax.set_title('S-P Method\nNumber of events per station',
                     fontdict={"size":18})
        ax.set_xlabel('Stations',fontdict={"size":14})
        ax.set_ylabel('Events',fontdict={"size":14})
        
        # Add the total number of ev_ids in a text box
        total_ev_ids = station_counts.sum()
        text_str = f"Total events: {total_ev_ids}"
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.5))
        
        ax.grid(True, linestyle='--', alpha=0.7) 
        
        plt.xticks(rotation=90, fontsize=14)  # Rotate x labels for readability
        plt.yticks(fontsize=14)  # Rotate x labels for readability
        plt.tight_layout()  # Adjust layout to avoid cutting off labels
        plt.show()
        # print(self.catalog)
    
    
    
    



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
    