import pandas as pd
import os
import string
import matplotlib.pyplot as plt
from delaware.core.enverus import plot_velocity_logs,well_fig

mydata = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/wells_aoi_all.csv"
formations = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/env_csv-FormationTops-332ba_2024-12-23.csv"
stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/delaware_onlystations_160824.csv"
output_fig = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/vp/vp.png"

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


# cond = (formations["API_UWI_12"] == form_r1[:-3]) |\
#          (formations["API_UWI_12"] == form_r2[:-3]) |\
#          (formations["API_UWI_12"] == form_r3[:-3] )
# formations = formations[cond]


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
        axes[col].text(text_loc[0], text_loc[1], 
                f"{string.ascii_lowercase[n]})", 
                horizontalalignment='left', 
                verticalalignment="top", 
                transform=axes[ col].transAxes, 
                fontsize="large", 
                fontweight="normal",
                bbox=box)
        n += 1

precision = 0.1
ax_rl1,lg1_rl1,lg2_rl1 = well_fig(data_rl1,formations,form_rl1,
         ax=axes[0],
         ylims=(-2,6),
         xlims=(1.5,6.5),
         data_color="gray",
         form_linestyle = "dashed",
         smooth_interval=precision,
         )

ax_1,lg1_1,lg2_1 = well_fig(data1,formations,form_r1,
         ax=axes[1],
         ylims=(-2,6),
         xlims=(1.5,6.5),
         form_linestyle = "dashed",
         smooth_interval=precision,
         )
text_loc2 = [0.5, 0.98]
ax_1.text(text_loc2[0], text_loc2[1], 
         "R1", 
         horizontalalignment='left', 
         verticalalignment="top", 
         transform=ax_1.transAxes, 
         fontsize="large", 
         fontweight="normal",
         bbox=box)
ax_2,lg1_2,lg2_2 = well_fig(data2,formations,form_r2,
         ax=axes[2],
         ylims=(-2,6),
         xlims=(1.5,6.5),
         form_linestyle="dashed",
         smooth_interval=precision,
         )
ax_2.text(text_loc2[0], text_loc2[1], 
         "R2", 
         horizontalalignment='left', 
         verticalalignment="top", 
         transform=ax_2.transAxes, 
         fontsize="large", 
         fontweight="normal",
         bbox=box)
print("R3")
ax_3,lg1_3,lg2_3 = well_fig(data3,formations,form_r3,
         ax=axes[3],
         ylims=(-2,6),
         xlims=(1.5,6.5),
         form_linestyle="dashed",
         smooth_interval=precision,
         )
ax_3.text(text_loc2[0], text_loc2[1], 
         "R3", 
         horizontalalignment='left', 
         verticalalignment="top", 
         transform=ax_3.transAxes, 
         fontsize="large", 
         fontweight="normal",
         bbox=box)
ax_rr1,lg1_rr1,lg2_rr1 = well_fig(data_rr1,formations,form_rr1,
         ax=axes[4],
         ylims=(-2,6),
         xlims=(1.5,6.5),
         data_color="gray",
         form_linestyle = "dashed",
         smooth_interval=precision,
         )


# Sort and add legend for formations.
global_legend_handles = {k: v for d in [lg1_1,lg1_2,lg1_3] for k, v in d.items()}
fig.legend(handles=list(global_legend_handles.values()), loc="upper left",
           fontsize=10, ncol=4, bbox_to_anchor=(0.05, 0.2))
# Add velocity model legend to the figure.
fig.legend(handles=lg2_3, loc="upper left", fontsize=10, ncol=1, bbox_to_anchor=(0.8, 0.2))

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
