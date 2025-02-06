from delaware.core.event.events import get_texnet_high_resolution_catalog
from delaware.core.event.picks import read_picks
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from obspy import UTCDateTime
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import pandas as pd
import os
import matplotlib as mpl
import numpy as np

# Dates
x = (-104.84329,-103.79942)
y = (31.3961,31.91505)
result_label = "HighRes TexNet"
events_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust/origin.csv"
output = "/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview/aoi.csv"
sta_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview/aoi_stations.csv"

# #FOr comparison
x = (-104.6,-104.2)
y = (31.6,31.75)
fig_out = os.path.join(r"/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview", "map_zoom.jpg")

data = pd.read_csv(output,parse_dates=["origin_time"])
sta_data = pd.read_csv(sta_path)


fig = plt.figure(figsize=plt.rcParams["figure.figsize"]*np.array([1.5,1]))
box = dict(boxstyle='round', facecolor='white', alpha=1)
text_loc = [0.05, 0.92]
grd = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[1.5, 1], height_ratios=[1,1])
fig.add_subplot(grd[:, 0])
plt.plot(data["longitude"], data["latitude"], 'k.', markersize=6, alpha=0.6)
plt.axis("scaled")
plt.xlim(np.array(x)+np.array([-0.05,0.05]))
plt.ylim(np.array(y)+np.array([-0.05,0.05]))
# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()

plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.gca().set_prop_cycle(None)
plt.plot(sta_data["longitude"], sta_data["latitude"], 'r^', markersize=8, alpha=0.7, label="Stations")
plt.plot(x[0]-1, y[0]-1, 'k.', markersize = 5, label = f"{result_label}")#, rasterized=True)
plt.legend(loc="lower left")
plt.text(text_loc[0], text_loc[1], 'A', horizontalalignment='left', verticalalignment="top", 
         transform=plt.gca().transAxes, fontsize="large", fontweight="normal", bbox=box)


fig.add_subplot(grd[0, 1])
plt.plot(data["longitude"], data["depth"], 'k.', markersize=2, alpha=1.0, rasterized=True)
# plt.axis("scaled")
plt.xlim(np.array(x))
plt.ylim([0,12])
plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
plt.xlabel("Longitude (°)")
plt.ylabel("Depth (km)")
plt.gca().set_prop_cycle(None)
plt.plot(x[0]-10, 31, 'k.', markersize=10)
plt.text(text_loc[0], text_loc[1], 'B', horizontalalignment='left', verticalalignment="top", 
         transform=plt.gca().transAxes, fontsize="large", fontweight="normal", bbox=box)

fig.add_subplot(grd[1, 1])
plt.plot(data["latitude"], data["depth"], 'k.', markersize=2, alpha=1.0, rasterized=True)
# plt.axis("scaled")
plt.xlim(np.array(y))
plt.ylim([0,12])
plt.gca().invert_yaxis()
plt.xlabel("Latitude (°)")
plt.ylabel("Depth (km)")
plt.gca().set_prop_cycle(None)
plt.plot(y[0]-10, 31, '.', markersize=10)
plt.tight_layout()
plt.text(text_loc[0], text_loc[1], 'C', horizontalalignment='left', verticalalignment="top", 
         transform=plt.gca().transAxes, fontsize="large", fontweight="normal", bbox=box)


fig.savefig(fig_out , bbox_inches = "tight", dpi = 300)
plt.show()