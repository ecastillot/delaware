from delaware.core.database import load_dataframe_from_sqlite
from delaware.loc.inv import prepare_cat2inv
from delaware.loc.s_p import plot_times_by_station
import os
import pandas as pd
from delaware.core.read import EQPicks

version = "10102024"
dw_path = os.path.join("/home/emmanuel/ecastillo/dev/delaware",version)

root = os.path.join(dw_path,"data/loc/sheng_s_p")
author = "sheng"
proj = "EPSG:3857"
# Custom colors for the stations
custom_palette = {"PB04": "#26fafa", 
                  "PB16": "#2dfa26", 
                #   "PB28": "#ad16db", 
                #   "PB37": "#1a3be3", 
                #   "WB03": "#ffffff", 
                #   "SA02": "#f1840f", 
                #   "PB24": "#0ea024", 
                  }

stations_relpath = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/delaware_onlystations_160824.csv"
stations_path = os.path.join(dw_path,stations_relpath)
stations = pd.read_csv(stations_path)
stations_columns = ["network","station","latitude","longitude","elevation"]
stations = stations[stations_columns]


catalog_path = os.path.join(root,"z_50.0",author,"catalog","catalog_sp_method.db")
picks_path = os.path.join(root,"z_50.0",author,"catalog","picks_sp_method.db")
catalog = load_dataframe_from_sqlite(db_name=catalog_path)
picks = load_dataframe_from_sqlite(db_name=picks_path)
picks = picks[picks["station"].isin(list(custom_palette.keys()))]


stations_with_picks = list(set(picks["station"].to_list()))
order = stations.copy()
order = order[order["station"].isin(stations_with_picks)]
order = order.sort_values("longitude",ignore_index=True,ascending=True)
order = order.drop_duplicates(subset="station")
order = order["station"].to_list()

catalog = catalog.drop_duplicates("ev_id")

cat, picks = prepare_cat2inv(catalog,picks,cat_columns_level=0,attach_station=stations)
picks["ts-tp"] = picks["tt_S"] - picks["tt_P"]

picks['ts-tp'] = picks['ts-tp'].astype(float)
stats_by_station = picks.groupby('station')['ts-tp'].describe()
print(picks['ts-tp'].dtype)
# print(picks)
print(stats_by_station)

output = os.path.join(dw_path,"script/s_p/sheng_s_p.png")
plot_times_by_station(picks,order=order,
                      palette=custom_palette,
                      ylim=(0,2),
                      savefig=output
                    #   boxprops={"edgecolor": "red"},       # Red box border
                    #     whiskerprops={"color": "red"},      # Red whisker lines
                    #     capprops={"color": "red"},          # Red cap lines
                    #     medianprops={"color": "red"}  
                      )