import pandas as pd
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from delaware.core.database.database import load_from_sqlite
from project.sp.utils import prepare_sp_analysis,plot_times_by_station
import os
import pandas as pd
from vel import VelModel
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

version = "10102024"
dw_path = os.path.join("/home/emmanuel/ecastillo/dev/delaware",version)

catalog_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/data/fig_1/catalog_sp_method.db"
picks_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/data/fig_1/picks_sp_method.db"
stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/delaware_onlystations_160824.csv"

# root = os.path.join(dw_path,"data/loc/s_p")
author = "growclust"
proj = "EPSG:3857"
# Custom colors for the stations
custom_palette = {"PB36": "#26fafa", 
                  "PB35": "#2dfa26", 
                  "PB28": "#ad16db", 
                  "PB37": "#1a3be3", 
                  "WB03": "#ffffff", 
                  "SA02": "#f1840f", 
                  "PB24": "#0ea024", 
                  }
region ={"PB36": 1, 
                  "PB35": 1, 
                  "PB28": 1, 
                  "PB37": 1, 
                  "SA02": 2, 
                  "WB03": 3, 
                  "PB24": 3, 
                  }

stations = pd.read_csv(stations_path)
stations_columns = ["network","station","latitude","longitude","elevation"]
stations = stations[stations_columns]


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

picks["ts-tp"] = picks["tt_S"] - picks["tt_P"]
picks['ts-tp'] = picks['ts-tp'].astype(float)
picks = picks[picks["station"].isin(list(custom_palette.keys()))]
# print(picks)
# exit()
stats_by_station = picks.groupby('station')['ts-tp'].describe()
# print(picks)
# print(stats_by_station)
# print(stats_by_station.loc["PB24"])

# exit()
vel_path = os.path.join(dw_path,"data/vel/DB_model.csv")
data = pd.read_csv(vel_path)
db1d_velmodel = VelModel(data=data,dtm=-0.7,name="db1d")

vel_path = os.path.join(dw_path,"data/vel/DW_model.csv")
data = pd.read_csv(vel_path)
sheng_velmodel = VelModel(data=data,dtm=-0.7,name="Sheng")

z_values = np.linspace(0,12,100)
v_mean_db1d = [db1d_velmodel.get_average_velocity(phase_hint="P", zmax=z) \
                                                    for z in z_values]
v_mean_db1d = np.array(v_mean_db1d)
v_mean_sheng = [sheng_velmodel.get_average_velocity(phase_hint="P", zmax=z)\
                                                        for z in z_values]
v_mean_sheng = np.array(v_mean_sheng)

form_path = os.path.join(dw_path,"data/vel/formation.csv")
form = pd.read_csv(form_path)
form =  form[form["Depth (km)"] != ""]

fig,axes = plt.subplots(1,len(stats_by_station) ,
                             figsize=(12, 8),
                             sharey=True,
                             gridspec_kw={'wspace': 0.05})
vp = np.linspace(2,6,100)
vps = {1:1.72,2:1.68,3:1.68}
for i,station in enumerate(order):
    
    color = custom_palette[station]
    if color == "#ffffff":
        color = "k"
    
    t1 = stats_by_station.loc[station]["25%"]
    t2 = stats_by_station.loc[station]["75%"]
    r = region[station]
    z1 = vp*(t1/(vps[r]-1))
    z2 = vp*(t2/(vps[r]-1))
    zw1 = vp*(t1/(1.89-1))
    zw2 = vp*(t2/(1.89-1))

    # Plot horizontal lines and add labels
    for _, row in form.iterrows():
        depth = row["Depth (km)"]
        label = row["Unit"]
        if isinstance(label,str):
            label = "\n".join(label.split(" "))
            axes[i].axhline(y=depth, color='k', linestyle='--', linewidth=0.5)  # Horizontal line
            axes[-1].text(1.02, depth, label, color='black', fontsize=8,
                    ha='left', va='top', 
                    transform=axes[-1].get_yaxis_transform())  # Label
            
    axes[i].plot(v_mean_db1d, z_values, 'blue', 
            linewidth=1, linestyle='-', 
            label='DB1D')
    axes[i].plot(v_mean_sheng, z_values,'orange', linewidth=1.5, linestyle='-', 
            label='Sheng (2022)')
    axes[i].plot(vp,z1,color=color,linewidth=1.5)
    axes[i].plot(vp,z2,color=color,linewidth=1.5)
    axes[i].fill_between(vp, z1, z2, color=color, alpha=0.3)  # Color span between y1 and y2
    axes[i].fill_between(vp, zw1, zw2, color="gray", alpha=0.3)  # Color span between y1 and y2
    
    axes[i].text(0.2, 0.95, f"{station}", 
             color="black", fontsize=12, 
             ha='left', va='top', 
             transform=axes[i].transAxes,
             bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.3') 
            #  backgroundcolor="white"
             )
    
        
    
    axes[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    # axes[i].legend(loc='lower left')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')  # Set the label to an empty string
    axes[i].set_ylim(ymin=0, ymax=12)
    axes[i].set_xlim(xmin=2, xmax=6.5)
    axes[i].invert_yaxis()
    # axes[i].grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
axes[0].set_ylabel('Depth [km]', fontsize=14)

output_path = os.path.join(dw_path,"script/s_p/depths.png")
# fig.savefig(output_path, dpi=300, bbox_inches='tight')


plt.show()
# output = os.path.join(dw_path,"script/s_p/s_p.png")
# plot_times_by_station(picks,order=order,
#                       palette=custom_palette,
#                       ylim=(0,2),
#                       savefig=output
#                     #   boxprops={"edgecolor": "red"},       # Red box border
#                     #     whiskerprops={"color": "red"},      # Red whisker lines
#                     #     capprops={"color": "red"},          # Red cap lines
#                     #     medianprops={"color": "red"}  
#                       )