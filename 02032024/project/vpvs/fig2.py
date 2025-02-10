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

fig, axes = plt.subplots(3,2,figsize=(10, 6))
global_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/vpvs"
output_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/vpvs/fig_2.png"


custom_palette = {
                    # "PB35": "#26fafa", 
                #   "PB36": "#2dfa26", 
                  "PB28": "#ad16db", 
                #   "PB37": "#1a3be3", 
                  "SA02": "#f1840f", 
                  "WB03": "#ffffff", 
                #   "PB24": "#0ea024", 
                  }

# r_color = {
#            "5":"green","10":"blue",
#            "15":"purple","20":"black"}
r_color = {
           "5":"yellow","10":"orange",
           "15":"red","20":"purple"}

# Share x-axis per column
for col in range(2):  # Iterate over columns (0 and 1)
    for row in range(1, 3):  # Skip the first row, link the rest to it
        axes[row, col].sharex(axes[0, col])

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
        

for i,cluster in enumerate(clusters_r):
    path = os.path.join(global_path,f"r{cluster}.csv")
    cat = pd.read_csv(path)

    Q1 = cat['v_ij'].quantile(0.10)
    Q3 = cat['v_ij'].quantile(0.90)
    iqr_cat = cat[(cat['v_ij'] >= Q1) & (cat['v_ij'] <= Q3)]
    
    ax = plot_vij_histogram(iqr_cat,cluster,
                       bins= np.arange(1.4,1.9,0.025),
                    ax=axes[i][0],
                    max="red",
                    #    output=output
                    )
    text_loc = [0.05, 0.2]
    ax.text(text_loc[0], text_loc[1], 
                f"R{cluster}", 
                horizontalalignment='left', 
                verticalalignment="top", 
                transform=ax.transAxes, 
                fontsize="medium", 
                fontweight="normal",
                bbox=box)

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
            
        ax= plot_vij_histogram_station(iqr_data,color=r_color[r],
                                   ax=axes[n][1],
                                   max=max)
        text_loc = [0.05, 0.2]
        ax.text(text_loc[0], text_loc[1], 
                    f"{station}", 
                    horizontalalignment='left', 
                    verticalalignment="top", 
                    transform=ax.transAxes, 
                    fontsize="medium", 
                    fontweight="normal",
                    bbox=box)
    
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