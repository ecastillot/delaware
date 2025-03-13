import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

color_regions = {1:"magenta",2:"blue",3:"green"}

stations = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/delaware_onlystations_160824.csv")


cross_plot_data = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/basement/cross_plot_data_31.7_-104.8_31.7_-103.8.csv")  # Update with actual file path
cross_elv_data = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/basement/cross_elevation_plot_data_31.7_-104.8_31.7_-103.8.csv")  # Update with actual file path
cross_elv_data["Elevation"] = cross_elv_data["Elevation"]*-1/1e3

# Load your DataFrame (replace this with your actual DataFrame)
df = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/02032024/project/reloc_depth/reloc_events.csv",parse_dates=["origin_time"])  # Replace with actual loading method if necessary

df["origin_time_numeric"] = (df["origin_time"] - df["origin_time"].min()).dt.total_seconds()
df["z_ori_from_surface"] = df["z_ori"] + np.interp(df["longitude"], 
                                                   cross_elv_data["Longitude"], 
                                                   cross_elv_data["Elevation"])
df["z_new_from_surface"] = df["z_new"] + np.interp(df["longitude"], 
                                                   cross_elv_data["Longitude"], 
                                                   cross_elv_data["Elevation"])

color_regions = {1:"magenta",2:"blue",3:"green"}
station_regions = {"PB35":1,"PB36":1,"PB28":1,"PB37":1,"SA02":2,"WB03":3,"PB24":3}
stations = stations[stations["station"].isin(list(station_regions.keys()))]
stations["region"] = stations["station"].apply(lambda x: station_regions[x])
stations["color"] = stations["region"].apply(lambda x: color_regions[x])
stations["elevation"] = np.interp(stations["longitude"], 
                                                   cross_elv_data["Longitude"], 
                                                   cross_elv_data["Elevation"])


from_surface = True
if from_surface:
        add = "_from_surface"
else:
        add = ""


highres = df.copy()
sp = df[df["Author_new"]=="S-P Depth Reloc"]
reloc = df[df["Author_new"]!="S-P Depth Reloc"]

highres.rename(columns={"z_ori"+add:"depth"+add}, inplace=True)
sp.rename(columns={"z_new"+add:"depth"+add}, inplace=True)
reloc.rename(columns={"z_new"+add:"depth"+add}, inplace=True)



fig,ax = plt.subplots(figsize=(14, 8))

sns.kdeplot(x="longitude", y="depth_from_surface",  
            data=sp[sp["region"]==1], ax=ax,
            color="magenta", fill=True, alpha=0.3)
sns.kdeplot(x="longitude", y="depth_from_surface",  
            data=sp[(sp["region"]==2) & (sp["depth_from_surface"]>4)], ax=ax,
            color="blue", fill=True, alpha=0.3)
sns.kdeplot(x="longitude", y="depth_from_surface",  
            data=sp[sp["region"]==3], ax=ax,
            color="green", fill=True, alpha=0.3)

# Plot Longitude vs Depth on the single axis
ax.plot(cross_elv_data["Longitude"], cross_elv_data["Elevation"] ,
        color="black", linestyle="-",
        label="Elevation", 
        # linewidth=2
        )
ax.plot(cross_plot_data["Longitude"], 
             cross_elv_data["Elevation"] + cross_plot_data["Elevation"], 
        color="red", linestyle="--",
        label="Basement")


sns.scatterplot(x="longitude", y="depth_from_surface", color="gray", 
                data=highres, label="TexNet HighRes", ax=ax, s=20, alpha=0.5)

df['timestamp'] = df['origin_time'].astype(np.int64)  # Convert to int timestamps

# Create colormap
norm = plt.Normalize(df['timestamp'].min(), df['timestamp'].max())
cmap = plt.cm.viridis
sc = ax.scatter(df['longitude'], df['z_new_from_surface'], 
                     c=df['timestamp'],
                     cmap=cmap, norm=norm)

ax.scatter(stations["station_longitude"], stations["elevation"], 
                color=stations["color"], s=100, marker="^", label="Stations")


# Colorbar with formatted date labels
cbar = plt.colorbar(sc, ax=ax)
tick_labels = pd.to_datetime(np.linspace(df['timestamp'].min(), df['timestamp'].max(), num=5))
cbar.set_ticks(tick_labels.astype(np.int64))
cbar.set_ticklabels(tick_labels.strftime('%Y-%m-%d'))
cbar.set_label("Origin Time")

# Set labels and title for the plot
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Depth", fontsize=12)
ax.set_xlim(-104.8, -103.8)
ax.set_ylim(15, -2)  # Invert the y-axis for depth
ax.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)  # Major grid
ax.set_title("Depth vs Longitude", fontsize=14)
ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Depth", fontsize=14)
ax.tick_params(axis='both', labelsize=14)

ax.set_xticklabels(ax.get_xticklabels(), ha='right')  # Rotate labels by 45 degrees


# Adjust opacity for individual hues
handles, labels = ax.get_legend_handles_labels()  # Get handles and labels from the legend
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
legend = ax.legend(loc="lower left", fontsize=14,markerscale=2)

for handle, text, label in zip(legend.legend_handles, legend.get_texts(), labels):
    if label == "Stations":
        handle.set_sizes([50])  # Reduce marker size (adjust value as needed)
        
        
# Adjust layout to avoid overlap
plt.tight_layout()

# Save the plot
fig.savefig("/home/emmanuel/ecastillo/dev/delaware/02032024/project/reloc_depth/fig_time.png")

# Show the plot
plt.show()
