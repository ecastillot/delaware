import pandas as pd
import os
import string
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from delaware.core.enverus import plot_velocity_logs,well_fig

mydata = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/wells_aoi_all.csv"
formations = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/env_csv-FormationTops-332ba_2024-12-23.csv"
stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/delaware_onlystations_160824.csv"
output_fig = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/vp/vp_ok.png"
output_folder_mean = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/vp"

data = pd.read_csv(mydata)
formations = pd.read_csv(formations)

form_rl1 = "42-109-31395-00-00"
form_r1 = "42-109-31382-00-00"
form_r2 = "42-109-30824-00-00"
form_r3 = "42-109-31383-00-00"
form_rr1 = "42-389-32460-00-00"

row_rl1 = "42-109-31395-00-00"
row_r1 = "42-109-00406-00-00"
row_r2 = "42-109-32455-00-00"
row_r3 = "42-109-31383-00-00"
row_rr1 = "42-389-32460-00-00"

well_rl1 = ["42-109-31395-00-00"]
well_r1 = ["42-109-00406-00-00"]
well_r2 = ["42-109-32255-00-00","42-109-32455-00-00"]
well_r3 = ["42-109-31383-00-00","42-109-31362-00-00",
           "42-109-31375-00-00","42-389-33816-00-00",
           "42-389-32876-00-00",
         #   "42-389-32460-00-00"
           ]
well_rr1 = ["42-389-32460-00-00"]

data_rl1 = data[data["well_name"].isin(well_rl1)]
data_rr1 = data[data["well_name"].isin(well_rr1)]
data1 = data[data["well_name"].isin(well_r1)]
data2 = data[data["well_name"].isin(well_r2)]
data3 = data[data["well_name"].isin(well_r3)]

##################################stratigraphy
strata = {
    "Delaware Mountain Group": [
        "Brushy Canyon",
        "Cherry Canyon",
        "Bell Canyon"
    ],
    "Bone Spring Group": [
        "Avalon Upper",
        "Avalon Middle",
        "Avalon Lower",
        "1st Bone Spring",
        "2nd Bone Spring",
        "2nd Bone Spring Sand",
        "3rd Bone Spring",
        "3rd Bone Spring Sand"
    ],
    "Wolfcamp Group": [
        "Wolfcamp A",
        "Wolfcamp A Lower",
        "Wolfcamp B Upper",
        "Wolfcamp B Lower",
        "Wolfcamp C",
        "Wolfcamp D",
        "Wolfcamp XY",
        "Wolfcamp Base"
    ],
    "Pennsylvanian Group": [
        "Penn Shale",
        "Penn Lower"
    ],
    "Mississippian Group": [
        "Miss Lime"
    ],
    "Woodford Group": [
        "Woodford",
        "Woodford Base"
    ]
}
# Choose a better spaced colormap: Set1 + Dark2 (as fallback)
set1 = plt.cm.get_cmap('Set1').colors  # 9 very distinct colors
dark2 = plt.cm.get_cmap('Dark2').colors  # 8 darker distinct colors
combined_colors = list(set1) + list(dark2)
color_list = [mcolors.to_hex(c) for c in combined_colors]

# New structure
formation_dict = {}

for i, (group, form) in enumerate(strata.items()):
    color = color_list[i % len(color_list)]
    for formation in form:
        formation_lowercase = formation.lower()
        formation_dict[formation_lowercase] = {
            "data": group,
            "color": color
        }
stratigraphy = formation_dict

#####################################333
# print(stratigraphy)
# exit()

fig, axes = plt.subplots(1, 5, 
                        sharey=True, 
                        figsize=(12, 8))

rows, cols = 1,axes.shape[0]  # Get the number of rows and columns
n = 0
for col in range(cols):  # Iterate over columns first
    for row in range(rows):  # Then iterate over rows
        box = dict(boxstyle='round', 
                    facecolor='white', 
                    alpha=1)
        text_loc = [0.05, 0.99]
        axes[col].annotate(f"({string.ascii_lowercase[n]})",
                           xy=(-0.1, 1.05),  # Outside top-left in axes coordinates
                           xycoords='axes fraction',
                           ha='left',
                           va='bottom',
                           fontsize="large",
                           fontweight="normal"
        )
        n += 1

text_loc2 = [5.9, -1.75]

precision = 0.1
ax_rl1,lg1_rl1,lg2_rl1 = well_fig(data_rl1,formations,form_rl1,
         ax=axes[0],
        stratigraphy=stratigraphy,
         ylims=(-2,6),
         xlims=(1.5,6.5),
         data_color="gray",
         form_linestyle = "dashed",
         smooth_interval=precision,
         
         )
# exit()
marker_0 = ax_rl1.plot(text_loc2[0], text_loc2[1],
          marker='P', markersize=16,
             linestyle='None', color='#c859b2',
             label="0")

ax_1,lg1_1,lg2_1 = well_fig(data1,formations,form_r1,
         stratigraphy=stratigraphy,
         ax=axes[1],
         ylims=(-2,6),
         xlims=(1.5,6.5),
         form_linestyle = "dashed",
         smooth_interval=precision,
         output_mean=os.path.join(output_folder_mean,"R1.csv")
         )

marker_1 = ax_1.plot(text_loc2[0], text_loc2[1],
          marker='P', markersize=16,
             linestyle='None', color='#10f60d',
             label="1")

ax_2,lg1_2,lg2_2 = well_fig(data2,formations,form_r2,
         stratigraphy=stratigraphy,
         ax=axes[2],
         ylims=(-2,6),
         xlims=(1.5,6.5),
         form_linestyle="dashed",
         smooth_interval=precision,
         output_mean=os.path.join(output_folder_mean,"R2.csv")
         )
marker_2 = ax_2.plot(text_loc2[0], text_loc2[1],
          marker='P', markersize=16,
             linestyle='None', color='#866ce4',
             label="2")

# print("R3")
ax_3,lg1_3,lg2_3 = well_fig(data3,formations,form_r3,
         stratigraphy=stratigraphy,
         ax=axes[3],
         ylims=(-2,6),
         xlims=(1.5,6.5),
         form_linestyle="dashed",
         smooth_interval=precision,
         output_mean=os.path.join(output_folder_mean,"R3.csv")
         )
marker_3 = ax_3.plot(text_loc2[0], text_loc2[1],
          marker='P', markersize=16,
             linestyle='None', color='#17e6ed',
             label="3")

ax_rr1,lg1_rr1,lg2_rr1 = well_fig(data_rr1,formations,form_rr1,
         stratigraphy=stratigraphy,
         ax=axes[4],
         ylims=(-2,6),
         xlims=(1.5,6.5),
         data_color="gray",
         form_linestyle = "dashed",
         smooth_interval=precision,
         )
marker_4 = ax_rr1.plot(text_loc2[0], text_loc2[1],
          marker='P', markersize=16,
             linestyle='None', color='#ffe415',
             label="4")

# Sort and add legend for formations.
global_legend_handles = {k: v for d in [lg1_1,lg1_2,lg1_3] for k, v in d.items()}
#ordering legends
global_legend_handles = {k: global_legend_handles[k] for k in strata if k in global_legend_handles}
fig.legend(handles=list(global_legend_handles.values()), 
           loc="upper left",title="Formations",
           fontsize=10, ncol=3, bbox_to_anchor=(0.05, 0.2))


log_handles = [marker_0[0],marker_1[0], marker_2[0], marker_3[0], marker_4[0]]
labels = ['0', '1','2','3','4']
fig.legend(handles=log_handles, loc="upper left", 
           labels=labels,
           title="Sonic Logs",
           fontsize=10, ncol=3, bbox_to_anchor=(0.61, 0.2))

# Add velocity model legend to the figure.
fig.legend(handles=lg2_3, loc="upper left", 
           title="Velocity Model",
           fontsize=10, ncol=1, bbox_to_anchor=(0.8, 0.2))

# Finalize layout and save or display the plot.
fig.tight_layout()
fig.subplots_adjust(bottom=0.26)
fig.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.show()

# stations = pd.read_csv(stations_path)
# region = [-104.843290,-103.799420,
#             31.396100,31.915050]
# print(data)
# plot_velocity_logs(data,depth="Depth[km]",
#                        ylims=(-2,6),
#                        xlims=(1.5,6.5),
#                     #    wells=well_r3,
#                      #   wells=well_r2,
#                        wells=well_r3,
#                        smooth_interval=0.1,
#                         scale_bar=50,
#                        region=region,
#                        stations=stations,
#                        formations=formations,
#                     #    wells = ["42-109-31362-00-00"],
#                     #    savefig=savefig,
#                        show=True)
# # plot_velocity_logs(data,depth="Depth[km]",
# #                     ylims=(-2,6),
# #                     wells=well_r1,
# #                     xlims=(1.5,6.5),
# #                     smooth_interval=0.1,
# #                     formations=formations)
# # plot_velocity_logs(data,depth="TVD[km]",ylims=(-2,6))
