import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec

color_regions = {1:"magenta",2:"blue",3:"green"}

stations = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/delaware_onlystations_160824.csv")


cross_plot_data = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/basement/cross_plot_data_31.7_-104.8_31.7_-103.8.csv")  # Update with actual file path
cross_elv_data = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/basement/cross_elevation_plot_data_31.7_-104.8_31.7_-103.8.csv")  # Update with actual file path
cross_elv_data["Elevation"] = cross_elv_data["Elevation"]*-1/1e3

# elev =  np.interp(cross_plot_data["Longitude"], 
#                     cross_elv_data["Longitude"], 
#                     cross_elv_data["Elevation"]) 
# cross_plot_data["Elevation"] = elev + abs(cross_plot_data["Elevation"])
# print(elev)
# print(abs(cross_plot_data["Elevation"]))

# Load your DataFrame (replace this with your actual DataFrame)
df = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/02032024/project/reloc_depth/reloc_events.csv")  # Replace with actual loading method if necessary


color_regions = {1:"magenta",2:"blue",3:"green"}
station_regions = {"PB35":1,"PB36":1,"PB28":1,"PB37":1,"SA02":2,"WB03":3,"PB24":3}
stations = stations[stations["station"].isin(list(station_regions.keys()))]
stations["region"] = stations["station"].apply(lambda x: station_regions[x])
stations["color"] = stations["region"].apply(lambda x: color_regions[x])
stations["elevation"] = np.interp(stations["longitude"], 
                                                   cross_elv_data["Longitude"], 
                                                   cross_elv_data["Elevation"])


from_sea_level = True
if from_sea_level:
    z_label = "_from_sea_level"
else:
    z_label = "_from_surface"

highres = df.copy()
sp = df[df["Author_new"]=="S-P Depth Reloc"].copy()
reloc = df[df["Author_new"]!="S-P Depth Reloc"].copy()


highres.rename(columns={"z_ori"+z_label:"depth"+z_label}, inplace=True)
sp.rename(columns={"z_new"+z_label:"depth"+z_label}, inplace=True)
reloc.rename(columns={"z_new"+z_label:"depth"+z_label}, inplace=True)



fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(1, 10, figure=fig)  # 2 rows, 3 columns
gs.update(wspace = 0.3, hspace = 2)

axes = []
axes.append(fig.add_subplot(gs[0, 0:8]))  
axes.append(fig.add_subplot(gs[0, 8:10], sharey=axes[0]))  

# First axis

sns.kdeplot(x="longitude", y="depth_from_sea_level",  
            data=sp[sp["region"]==1], ax=axes[0],
            color="magenta", fill=True, alpha=0.3)
sns.kdeplot(x="longitude", y="depth_from_sea_level",  
            data=sp[(sp["region"]==2) & (sp["depth_from_sea_level"]>4)], ax=axes[0],
            color="blue", fill=True, alpha=0.3)
sns.kdeplot(x="longitude", y="depth_from_sea_level",  
            data=sp[sp["region"]==3], ax=axes[0],
            color="green", fill=True, alpha=0.3)

# Plot Longitude vs Depth on the single axis
axes[0].plot(cross_elv_data["Longitude"], cross_elv_data["Elevation"] ,
        color="black", linestyle="-",
        label="Elevation", 
        # linewidth=2
        )
axes[0].plot(cross_plot_data["Longitude"], cross_elv_data["Elevation"] + cross_plot_data["Elevation"], 
        color="red", linestyle="--",
        label="Basement")

# print(df)
# exit()
sns.scatterplot(x="longitude", y="depth_from_sea_level", color="gray", 
                data=highres, label="TexNet HighRes", ax=axes[0], s=20, alpha=0.5)
sns.scatterplot(x="longitude", y="depth_from_sea_level", color="darkorange", marker="+",
                data=reloc, 
                label="Relative Reloc from S-P Depth Reloc", 
                ax=axes[0], 
                s=20,
                # alpha=0.2
                )
sns.scatterplot(x="longitude", y="depth_from_sea_level", color="darkorange", 
                edgecolor="red",linewidth=0.25,
                data=sp, label="S-P Depth Reloc", ax=axes[0], s=20, alpha=0.2)

axes[0].scatter(stations["station_longitude"], stations["elevation"], 
                color=stations["color"], s=100, marker="^", label="Stations")


# Set labels and title for the plot
axes[0].set_xlabel("Longitude", fontsize=12)
axes[0].set_ylabel("Depth", fontsize=12)
axes[0].set_xlim(-104.8, -103.8)
axes[0].set_ylim(15, -2)  # Invert the y-axes[0]is for depth
axes[0].grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)  # Major grid
axes[0].set_title("Depth vs Longitude", fontsize=14)
axes[0].set_xlabel("Longitude", fontsize=14)
axes[0].set_ylabel("Depth", fontsize=14)
axes[0].tick_params(axis='both', labelsize=14)

axes[0].set_xticklabels(axes[0].get_xticklabels(), ha='right')  # Rotate labels by 45 degrees

axes[1].hist(highres["depth_from_sea_level"],
             bins=50, color="gray",
            #  histtype="step",
             alpha=0.5, 
             orientation="horizontal",density=True)
axes[1].hist(df["z_new_from_sea_level"], bins=50, color="darkorange",
            #  histtype="step",
             alpha=0.5, 
             orientation="horizontal",density=True)

axes[1].hist(sp[sp["region"]==1]["depth_from_sea_level"], bins=20, color="magenta",
             histtype="step",
             alpha=0.7, label="Region 1",
             orientation="horizontal",density=True)
axes[1].hist(sp[sp["region"]==2]["depth_from_sea_level"], bins=20, color="blue",
             histtype="step",label="Region 2",
             alpha=0.7, 
             orientation="horizontal",density=True)
axes[1].hist(sp[sp["region"]==3]["depth_from_sea_level"], bins=20, color="green",
             histtype="step",label="Region 3",
             alpha=0.7, 
             orientation="horizontal",density=True)

# Optional: Hide redundant y-axis labels for the second subplot
axes[1].tick_params(labelleft=False, labelsize=14)
axes[1].set_xlabel("Density", fontsize=14)
axes[1].legend( fontsize=10)
# axes[1].legend( fontsize=14)
# axes[1].set_ylim(15, -2)  # Invert the y-axes[0]is for depth


# Adjust opacity for individual hues
handles, labels = axes[0].get_legend_handles_labels()  # Get handles and labels from the legend
for handle, label in zip(handles, labels):
    if label == "S-P Depth Reloc":  # Apply higher opacity for the gray points
        handle.set_alpha(1)  # Set full opacity (gray)
    elif label == "TexNet HighRes":  # Apply lower opacity for the orange points
        handle.set_alpha(0.7)  # Set lower opacity (orange)
    elif label == "Stations":
        handle.set_alpha(1)
    else:
        print(label)
        handle.set_alpha(1)
legend = axes[0].legend(loc="lower left", fontsize=14,markerscale=2)

for handle, text, label in zip(legend.legend_handles, legend.get_texts(), labels):
    if label == "Stations":
        # text.set_fontsize(10)  # Decrease text size
        handle.set_sizes([50])  # Reduce marker size (adjust value as needed)
    elif label == "S-P Depth Reloc":
        handle.set_linewidth(2)  # Increase line thickness (adjust as needed)
    elif label == "Relative Reloc from S-P Depth Reloc":
        handle.set_linewidth(2)  # Increase line thickness (adjust as needed)
# Adjust layout to avoid overlap
plt.tight_layout()

# Save the plot
fig.savefig("/home/emmanuel/ecastillo/dev/delaware/02032024/project/reloc_depth/fig_depth_sea_level.png")

# Show the plot
plt.show()
