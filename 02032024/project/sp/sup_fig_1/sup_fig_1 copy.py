import pandas as pd
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from delaware.core.database.database import load_from_sqlite
from project.sp.utils import prepare_sp_analysis,plot_times_by_station

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
                  }

all_stations = pd.read_csv(stations_path)
fig,axes = plt.subplots(2,3,figsize=((12,8)))

positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]  # Upper row is centered, lower row full

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

  # row = n // 3  # Calculate row index (integer division)
  # col = n % 3   # Calculate column index (remainder)
  row, col = positions[n]  # Get the specific subplot position

  ax = plot_times_by_station(picks,order=order,
                        palette=general_palette,
                        ylim=(0,2),
                        show=False,
                        ax=axes[row, col]  # Assign the correct subplot
                        )
    



# fig.subplots_adjust(bottom=0.2)
# arrow = mpatches.FancyArrow(0.15, 0.05, 0.8, 0, width=0.001, transform=fig.transFigure, color="black")
# fig.patches.append(arrow)

# fig.text(0.15, 0.07, "W", ha="center", va="center", fontsize=12, fontweight="bold")
# fig.text(0.95, 0.07, "E", ha="center", va="center", fontsize=12, fontweight="bold")

# fig.savefig(output_path, dpi=300, bbox_inches="tight")
# Hide the empty subplot (at position [0,2])
axes[0, 0].axis("off")
plt.tight_layout()  # Improve spacing
plt.show()



