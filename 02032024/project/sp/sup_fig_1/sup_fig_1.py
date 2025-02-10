import pandas as pd
import os
import string
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from delaware.core.database.database import load_from_sqlite
from project.sp.utils import prepare_sp_analysis,sup_fig_1

stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/delaware_onlystations_160824.csv"
output_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/sup_fig_1/fig_1.png"
radii = [1,2,3,4,5]

custom_palette = {"PB35": "#26fafa", 
                  "PB36": "#2dfa26", 
                  "PB28": "#ad16db", 
                  "PB37": "#1a3be3", 
                  "WB03": "#ffffff", 
                  "SA02": "#f1840f", 
                  "PB24": "#0ea024", 
                  "PB24": "#0ea024", 
                  "PB24": "#0ea024", 
                  "PB04": "red", 
                  "PB16": "red", 
                  }

all_stations = pd.read_csv(stations_path)

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(6, 6, figure=fig)  # 2 rows, 3 columns
gs.update(wspace = 0.3, hspace = 2)

# Define axes with correct positioning
axes = []
axes.append(fig.add_subplot(gs[1:3, 0:2]))  
axes.append(fig.add_subplot(gs[3:5, 0:2]))  
axes.append(fig.add_subplot(gs[0:2, 2:6])) 
axes.append(fig.add_subplot(gs[2:4, 2:6]))  
axes.append(fig.add_subplot(gs[4:6, 2:6]))  

# plt.show()
# exit()

for n,r in enumerate(radii):
  print(r)
  catalog_path = f"/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/data/sup_fig_1/{r}_km/catalog_sp_method.db"
  picks_path = f"/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/data/sup_fig_1/{r}_km/picks_sp_method.db"


  picks = load_from_sqlite(picks_path)
  catalog = load_from_sqlite(catalog_path)
  
  stations_columns = ["network","station","latitude","longitude","elevation"]
  stations = all_stations[stations_columns]

  stations_with_picks = list(set(picks["station"].to_list()))
  order = stations.copy()
  order = order[order["station"].isin(stations_with_picks)]
  order = order.sort_values("longitude",ignore_index=True,ascending=True)
  order = order.drop_duplicates(subset="station")
  order = order["station"].to_list()




  catalog,picks = prepare_sp_analysis(catalog,picks,cat_columns_level=0)

  picks["ts-tp"] = picks["tt_S"] - picks["tt_P"]
  picks['ts-tp'] = picks['ts-tp'].astype(float)
  stats_by_station = picks.groupby('station')['ts-tp'].describe()

  general_palette = dict([(x,"gray") for x in order])

  for key,value in custom_palette.items():
    general_palette[key] = value


  ax = sup_fig_1(picks,order=order,
                        palette=general_palette,
                        ylim=(0,2),
                        show=False,
                        ax=axes[n]  # Assign the correct subplot
                        )
  ax.set_xlabel("")
  ax.set_title(f"{r} km",
                  fontdict={"size":10,
                            "weight":"bold"})
  
  if n>1:
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    text_loc = [0.025, 0.92]
  else:
    text_loc = [0.05, 0.92]
    

  box = dict(boxstyle='round', 
             facecolor='white', 
             alpha=1)
  ax.text(text_loc[0], text_loc[1], 
          f"{string.ascii_lowercase[n]})", 
          horizontalalignment='left', 
          verticalalignment="top", 
         transform=ax.transAxes, 
         fontsize="large", 
         fontweight="normal",
         bbox=box)


# fig.subplots_adjust(bottom=0.2)
# arrow = mpatches.FancyArrow(0.15, 0.05, 0.8, 0, width=0.001, transform=fig.transFigure, color="black")
# fig.patches.append(arrow)

# fig.text(0.15, 0.07, "W", ha="center", va="center", fontsize=12, fontweight="bold")
# fig.text(0.95, 0.07, "E", ha="center", va="center", fontsize=12, fontweight="bold")

# fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.tight_layout()  # Improve spacing
plt.show()



