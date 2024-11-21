from delaware.core.database import load_dataframe_from_sqlite
from delaware.loc.inv import prepare_cat2inv
from delaware.loc.s_p import plot_times_by_station
import os
import pandas as pd

version = "10102024"
dw_path = os.path.join("/home/emmanuel/ecastillo/dev/delaware",version)

root = os.path.join(dw_path,"data/loc/s_p")
author = "growclust"
proj = "EPSG:3857"

stations_relpath = "data_git/stations/standard_stations.csv"
stations_path = os.path.join(dw_path,stations_relpath)
stations = pd.read_csv(stations_path)
stations_columns = ["network","station","latitude","longitude","elevation","x[km]","y[km]"]
stations = stations[stations_columns]


catalog_path = os.path.join(root,"z_10.0",author,"catalog","catalog_sp_method.db")
picks_path = os.path.join(root,"z_10.0",author,"catalog","picks_sp_method.db")
catalog = load_dataframe_from_sqlite(db_name=catalog_path)
picks = load_dataframe_from_sqlite(db_name=picks_path)

stations_with_picks = list(set(picks["station"].to_list()))
order = stations.copy()
order = order[order["station"].isin(stations_with_picks)]
order = order.sort_values("longitude",ignore_index=True,ascending=True)
order = order.drop_duplicates(subset="station")
order = order["station"].to_list()
print(order)

cat, picks = prepare_cat2inv(catalog,picks,cat_columns_level=0,attach_station=stations)
picks["ts-tp"] = picks["tt_S"] - picks["tt_P"]

# Custom colors for the stations
custom_palette = {"PB36": "#26fafa", 
                  "PB35": "#2dfa26", 
                  "PB28": "#fac226", 
                  "PB37": "#1a3be3", 
                  "WB03": "#ffffff", 
                  "SA02": "gray", 
                  "WB05": "gray", 
                  "PB24": "gray", 
                  }

output = os.path.join(dw_path,"script/s_p/s_p.png")
plot_times_by_station(picks,order=order,
                      palette=custom_palette,
                      savefig=output
                    #   boxprops={"edgecolor": "red"},       # Red box border
                    #     whiskerprops={"color": "red"},      # Red whisker lines
                    #     capprops={"color": "red"},          # Red cap lines
                    #     medianprops={"color": "red"}  
                      )
# print(cat,picks)