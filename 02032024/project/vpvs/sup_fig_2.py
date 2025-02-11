import matplotlib.pyplot as plt
import os
import string
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from project.vpvs.utils import plot_vij_histogram,plot_vij_histogram_station
import glob
from matplotlib.lines import Line2D


clusters_r = [1,2,3]

fig, axes = plt.subplots(3,3,figsize=(10, 6),sharey=True)
global_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/vpvs"
output_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/vpvs/sup_fig_2.png"


custom_palette = {
                  "PB37": "#1a3be3", 
                  "PB28": "#ad16db", 
                "PB35": "#26fafa", 
                  "PB36": "#2dfa26", 
                  "SA02": "#f1840f", 
                  "PB24": "#0ea024", 
                  "WB03": "#ffffff", 
                  "PB16": "red", 
                  "PB04": "red", 
                  }

# r_color = {
#            "5":"green","10":"blue",
#            "15":"purple","20":"black"}
r_color = {
           "5":"yellow","10":"orange",
           "15":"red","20":"purple"}


rows, cols = axes.shape  # Get the number of rows and columns
n = 0
for col in range(cols):  # Iterate over columns first
    for row in range(rows):  # Then iterate over rows
        box = dict(boxstyle='round', 
                    facecolor='white', 
                    alpha=1)
        text_loc = [0.05, 0.92]
        axes[row, col].text(text_loc[0], text_loc[1], 
                f"{string.ascii_lowercase[n]})", 
                horizontalalignment='left', 
                verticalalignment="top", 
                transform=axes[row, col].transAxes, 
                fontsize="large", 
                fontweight="normal",
                bbox=box)
        n += 1
        


for n,(key,val) in enumerate(custom_palette.items()):
    query = glob.glob(os.path.join(global_path,"stations",f"{key}*.csv"))
    for path in query:
        basename = os.path.basename(path).split(".")[0]
        station,r = basename.split("_")
        
        if r not in r_color.keys():
            continue
        
        data = pd.read_csv(path)
        Q1 = data['v_ij'].quantile(0.10)
        Q3 = data['v_ij'].quantile(0.90)
        iqr_data = data[(data['v_ij'] >= Q1) & (data['v_ij'] <= Q3)]
        
        if r=="15":
            max = "red"
        else:
            max = None
        
        row, col = divmod(n, 3)
        ax= plot_vij_histogram_station(iqr_data,color=r_color[r],
                                   ax=axes[col][row],
                                   max=max)
        # Set y-axis label only for the first column
        if col != 0:
            axes[row, col].set_ylabel("")
            # axes[row, col].tick_params(left=False)  # Hide y-ticks on other columns
            
        
        text_loc = [0.05, 0.2]
        ax.text(text_loc[0], text_loc[1], 
                    f"{station}", 
                    horizontalalignment='left', 
                    verticalalignment="top", 
                    transform=ax.transAxes, 
                    fontsize="medium", 
                    fontweight="normal",
                    bbox=box)
 
 
# axes[1, 2].set_visible(False)    
# axes[2, 2].set_visible(False)    

legend_elements = [Line2D([0], [0], color=color, 
                lw=2, label=f"{key} km") \
                    for key, color in r_color.items()]

legend_max = [Line2D([0], [0], color="red", 
                lw=2, linestyle="--",label=f"Max. Value")\
                    ]

fig.legend(handles=legend_elements, 
        #    loc='lower right', , 
        # loc='lower center',
           ncol=len(r_color), 
        #    loc=(0.5, 0.035),
           loc=(0.5, 1-0.1),
           frameon=False,title="Radius")

fig.legend(handles=legend_max, 
        #    loc='lower right', , 
        # loc='lower center',
           ncol=len(r_color), 
        #    loc=(0.25, 0.05),
           loc=(0.25, 1-0.1),
           frameon=False)

# plt.subplots_adjust(bottom=0.2)  # Adjust bottom spacing to fit the legend
plt.subplots_adjust(bottom=0.2)  # Adjust bottom spacing to fit the legend
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()