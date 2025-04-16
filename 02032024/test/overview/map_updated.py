# from delaware.core.event.events import get_texnet_high_resolution_catalog
# from delaware.core.event.picks import read_picks
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from obspy import UTCDateTime
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import pandas as pd
import os
import matplotlib as mpl
import numpy as np
import string

# Dates
x = (-104.84329,-103.79942)
y = (31.3961,31.91505)
result_label = "TexNet"
output = "/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview/aoi_original.csv"
sta_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview/aoi_stations.csv"

fig_out = os.path.join(r"/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview", "map_updated.jpg")


# #FOr comparison
# x = (-104.6,-104.2)
# y = (31.6,31.75)
# fig_out = os.path.join(r"/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview", "map_updated_zoom.jpg")

data = pd.read_csv(output,parse_dates=["origin_time"])
sta_data = pd.read_csv(sta_path)


fig = plt.figure(figsize=plt.rcParams["figure.figsize"] * np.array([1.5, 1]))
box = dict(boxstyle='round', facecolor='white', alpha=1)
text_loc = [0.05, 0.92]

grd = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[1.5, 1], height_ratios=[1, 1])

# Panel A: Map View
ax0 = fig.add_subplot(grd[:, 0])
ax0.plot(data["longitude"], data["latitude"], 'k.', markersize=6, alpha=0.6)
ax0.set_aspect('equal', adjustable='box')
ax0.set_xlim(np.array(x) + np.array([-0.05, 0.05]))
ax0.set_ylim(np.array(y) + np.array([-0.05, 0.05]))
ax0.set_xlabel("Longitude (°)")
ax0.set_ylabel("Latitude (°)")
ax0.set_prop_cycle(None)
ax0.plot(sta_data["longitude"], sta_data["latitude"], 'r^', markersize=8, alpha=0.7, label="Stations")
ax0.plot(x[0] - 1, y[0] - 1, 'k.', markersize=5, label=f"{result_label}")
ax0.legend(loc="lower left")

# Panel B: Longitude vs Depth
ax1 = fig.add_subplot(grd[0, 1])
ax1.plot(data["longitude"], data["depth"], 'k.', markersize=2, alpha=1.0, rasterized=True)
ax1.set_xlim(np.array(x))
ax1.set_ylim([0, 12])
ax1.invert_yaxis()
ax1.set_xlabel("Longitude (°)")
ax1.set_ylabel("Depth (km)")
ax1.set_prop_cycle(None)
ax1.plot(x[0] - 10, 31, 'k.', markersize=10)

# Panel C: Latitude vs Depth
ax2 = fig.add_subplot(grd[1, 1])
ax2.plot(data["latitude"], data["depth"], 'k.', markersize=2, alpha=1.0, rasterized=True)
ax2.set_xlim(np.array(y))
ax2.set_ylim([0, 12])
ax2.invert_yaxis()
ax2.set_xlabel("Latitude (°)")
ax2.set_ylabel("Depth (km)")
ax2.set_prop_cycle(None)
ax2.plot(y[0] - 10, 31, '.', markersize=10)

# Store all axes in a list (order matters!)
axes = [ax0, ax1, ax2]

# Auto-labeling with letters (a), (b), (c), ...
for n, ax in enumerate(axes):
    ax.annotate(f"({string.ascii_lowercase[n]})",
                xy=(-0.1, 1.05),  # Slightly outside top-left
                xycoords='axes fraction',
                ha='left',
                va='bottom',
                fontsize="large",
                fontweight="normal",
                # bbox=box
                )


fig.tight_layout()
fig.savefig(fig_out, bbox_inches="tight", dpi=300)
plt.show()