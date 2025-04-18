import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
import datetime as dt

# Function to load data
def load_data():
    stations = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/delaware_onlystations_160824.csv")
    cross_plot_data = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/basement/cross_plot_data_31.7_-104.8_31.7_-103.8.csv")
    cross_elv_data = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/basement/cross_elevation_plot_data_31.7_-104.8_31.7_-103.8.csv")
    cross_elv_data["Elevation"] = cross_elv_data["Elevation"] * -1 / 1e3  # Convert elevation to km

    df = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/02032024/project/reloc_depth/reloc_events.csv", parse_dates=["origin_time"])

    return df, cross_plot_data, cross_elv_data, stations

# Function to preprocess and prepare the data
def preprocess_data(df, cross_elv_data):
    df["origin_time_numeric"] = df['origin_time'].astype(np.int64)  # Seconds since the earliest event
    # df["z_ori_from_surface"] = df["z_ori"] + np.interp(df["longitude"], cross_elv_data["Longitude"], cross_elv_data["Elevation"])
    # df["z_new_from_surface"] = df["z_new"] + np.interp(df["longitude"], cross_elv_data["Longitude"], cross_elv_data["Elevation"])
    df['month'] = df['origin_time'].dt.to_period('M')
    return df.sort_values('month')

def preprocess_stations(stations, cross_elv_data):
    color_regions = {1:"magenta",2:"blue",3:"green"}
    station_regions = {"PB35": 1, "PB36": 1, "PB28": 1, "PB37": 1, "SA02": 2, "WB03": 3, "PB24": 3}
    stations = stations[stations["station"].isin(list(station_regions.keys()))]
    stations["region"] = stations["station"].apply(lambda x: station_regions[x])
    stations["color"] = stations["region"].apply(lambda x: color_regions[x])
    stations["elevation"] = np.interp(stations["longitude"], cross_elv_data["Longitude"], cross_elv_data["Elevation"])
    return stations

# Function to create the base plot
def create_base_plot(df, cross_elv_data, cross_plot_data, stations,starttime=None,endtime=None,
                     num_colors = 8):
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot Longitude vs Depth on the single axis
    ax.plot(cross_elv_data["Longitude"], cross_elv_data["Elevation"], color="black", linestyle="-", label="Elevation")
    # ax.plot(cross_plot_data["Longitude"], cross_elv_data["Elevation"] + cross_plot_data["Elevation"], color="red", linestyle="--", label="Basement")
    ax.plot(cross_plot_data["Longitude"], cross_plot_data["Elevation"], color="red", linestyle="--", label="Basement")

    # Plot stations on top of the static part of the plot
    ax.scatter(stations["station_longitude"], stations["elevation"], 
                color=stations["color"], s=100, marker="^", label="Stations")
    
    sc = ax.scatter([], [])  # Empty scatter for the colorbar
    # Set axis labels and other static plot properties
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
    
    # Add dynamic text annotation for the time frame (initially empty)
    time_text = ax.text(0.3, 0.05, "", transform=ax.transAxes,
                        fontsize=16, ha="right", va="bottom", 
                        color="black")
    
    # norm = plt.Normalize(df['origin_time_numeric'].min(), df['origin_time_numeric'].max())
    # cmap = plt.cm.rainbow
    if starttime is None:
        starttime = df['origin_time_numeric'].min()
    if endtime is None:
        endtime = df['origin_time_numeric'].max()
    
    cmap = plt.cm.rainbow
    cmap = cmap(np.arange(cmap.N))[::-1]  # Reverse the colormap
    cmap = mcolors.ListedColormap(cmap)
    
    bounds = np.linspace(starttime, endtime, num_colors + 1)  # Divide range into equal intervals
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    sc = ax.scatter([], [], 
                     c=[],
                     cmap=cmap, norm=norm)
    
    # Colorbar with formatted date labels
    cbar = plt.colorbar(sc, ax=ax, boundaries=bounds)
    tick_labels = pd.to_datetime(np.linspace(starttime, 
                                             endtime,
                                             num=num_colors+1))
    # print(tick_labels,starttime,endtime)
    # exit()
    cbar.set_ticks(tick_labels.astype(np.int64))
    # cbar.set_ticklabels(tick_labels.strftime('%Y-%m-%d'))
    cbar.set_ticklabels(tick_labels.strftime('%Y'),fontsize=14)
    cbar.set_label("Origin Time",fontsize=12)

    return fig, ax, cmap,norm, time_text

# Function to update the plot for each frame
def update(frame, df, sc, time_text):
    # Filter data for the current month
    current_month = df[df['month'] <= frame]
    
    # Update the scatter plot with new data points
    sc.set_offsets(np.column_stack([current_month['longitude'],
                                    current_month['z_new_from_surface']]))
    sc.set_array(current_month['origin_time_numeric'])  # Color points based on 'origin_time_numeric'
    
    # Update the dynamic text for the time frame
    current_date = current_month['origin_time'].max().strftime('%Y-%m')
    time_text.set_text(f"Time Frame: {current_date}")
    
    return sc,time_text

# Main function to create the animation
def animate(df, ax, frame_range,cmap,norm,time_text):
    # Create scatter plot for animated events
    sc = ax.scatter([], [], c=[], cmap=cmap,norm=norm)

    # Create the animation - one frame per month
    ani = FuncAnimation(fig, update, frames=frame_range, fargs=(df, sc,time_text), interval=1000, blit=False)

    return ani

# Load and preprocess the data
df, cross_plot_data, cross_elv_data, stations = load_data()
df = preprocess_data(df, cross_elv_data)
stations = preprocess_stations(stations, cross_elv_data)

# Create the base plot
fig, ax,cmap,norm,time_text = create_base_plot(df, cross_elv_data, cross_plot_data, stations,
                                     starttime=pd.to_datetime('2017-01-01').value,
                                     endtime=pd.to_datetime('2025-01-01').value)

# Create the animation
frame_range = df['month'].unique()  # Use unique months as frames
ani = animate(df, ax, frame_range,cmap,norm,time_text)

out = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/reloc_depth/seismic_animation.mp4"
ani.save(out, writer='ffmpeg', fps=2)  # Save as video
plt.show()

# sudo apt-get install ffmpeg
