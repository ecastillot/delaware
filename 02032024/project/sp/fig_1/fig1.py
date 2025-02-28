import pandas as pd
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from delaware.core.database.database import load_from_sqlite
from project.sp.utils import prepare_sp_analysis,plot_times_by_station

catalog_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/data/fig_1/catalog_sp_method.db"
picks_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/data/fig_1/picks_sp_method.db"
stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/delaware_onlystations_160824.csv"
output_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/fig_1/fig_1.png"

csv_output_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/fig_1/fig_1_picks.csv"

custom_palette = {"PB35": "#26fafa", 
                  "PB36": "#2dfa26", 
                  "PB28": "#ad16db", 
                  "PB37": "#1a3be3", 
                  "WB03": "#ffffff", 
                  "SA02": "#f1840f", 
                  "PB24": "#0ea024", 
                  }


picks = load_from_sqlite(picks_path)
catalog = load_from_sqlite(catalog_path)

stations = pd.read_csv(stations_path)
stations_columns = ["network","station","latitude","longitude","elevation"]
stations = stations[stations_columns]

stations = stations[stations["station"].isin(list(custom_palette.keys()))]

stations_with_picks = list(set(picks["station"].to_list()))
order = stations.copy()
order = order[order["station"].isin(stations_with_picks)]
order = order.sort_values("longitude",ignore_index=True,ascending=True)
order = order.drop_duplicates(subset="station")
order = order["station"].to_list()

# print(catalog.info(),picks.info())
catalog,picks = prepare_sp_analysis(catalog,picks,cat_columns_level=0)

print(picks[picks["tt_P"]<0])
exit()

picks["ts-tp"] = picks["tt_S"] - picks["tt_P"]
picks['ts-tp'] = picks['ts-tp'].astype(float)
picks.to_csv(csv_output_path,index=False)
# print(picks)
# exit()
stats_by_station = picks.groupby('station')['ts-tp'].describe()





fig,ax = plot_times_by_station(picks,order=order,
                      palette=custom_palette,
                      ylim=(0,2),
                      show=False,
                      # savefig=output_path
                    #   boxprops={"edgecolor": "red"},       # Red box border
                    #     whiskerprops={"color": "red"},      # Red whisker lines
                    #     capprops={"color": "red"},          # Red cap lines
                    #     medianprops={"color": "red"}  
                      )

# fig.annotate(
#     text="",
#     xy=(0, -0.15), xycoords='axes fraction',
#     xytext=(1,-0.15), textcoords='axes fraction',
#     arrowprops=dict(arrowstyle="<|-|>", lw=1.5, color='black'),
#     fontsize=12, ha='center', va='top'
# )
# ax.text(xmin+0.1,0.2,"W",fontdict={"size":16})
# ax.text(xmax-0.15,0.2,"E",fontdict={"size":16})


fig.subplots_adjust(bottom=0.2)
arrow = mpatches.FancyArrow(0.15, 0.05, 0.8, 0, width=0.001, transform=fig.transFigure, color="black")
fig.patches.append(arrow)

fig.text(0.15, 0.07, "W", ha="center", va="center", fontsize=12, fontweight="bold")
fig.text(0.95, 0.07, "E", ha="center", va="center", fontsize=12, fontweight="bold")

fig.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()



