import pandas as pd
import os
from delaware.core.database.database import load_from_sqlite
from utils import prepare_sp_analysis,plot_times_by_station

catalog_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/data_5km/catalog_sp_method.db"
picks_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/data_5km/picks_sp_method.db"
stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/delaware_onlystations_160824.csv"

output_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/s_p_5km.png"

picks = load_from_sqlite(picks_path)
catalog = load_from_sqlite(catalog_path,)

stations = pd.read_csv(stations_path)
stations_columns = ["network","station","latitude","longitude","elevation"]
stations = stations[stations_columns]

stations_with_picks = list(set(picks["station"].to_list()))
order = stations.copy()
order = order[order["station"].isin(stations_with_picks)]
order = order.sort_values("longitude",ignore_index=True,ascending=True)
order = order.drop_duplicates(subset="station")
order = order["station"].to_list()

# print(catalog.info(),picks.info())
catalog,picks = prepare_sp_analysis(catalog,picks,cat_columns_level=0)

picks["ts-tp"] = picks["tt_S"] - picks["tt_P"]
picks['ts-tp'] = picks['ts-tp'].astype(float)
stats_by_station = picks.groupby('station')['ts-tp'].describe()


custom_palette = {"PB04": "#26fafa", 
                  "PB36": "#2dfa26", 
                  "PB28": "#ad16db", 
                  "PB37": "#1a3be3", 
                  "WB03": "#ffffff", 
                  "SA02": "#f1840f", 
                  "PB24": "#0ea024", 
                  }

plot_times_by_station(picks,order=order,
                    #   palette=custom_palette,
                      ylim=(0,2),
                      savefig=output_path
                    #   boxprops={"edgecolor": "red"},       # Red box border
                    #     whiskerprops={"color": "red"},      # Red whisker lines
                    #     capprops={"color": "red"},          # Red cap lines
                    #     medianprops={"color": "red"}  
                      )



