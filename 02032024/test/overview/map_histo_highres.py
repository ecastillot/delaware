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
result_label = "TexNet"
output = "/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview/aoi.csv"
sta_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview/aoi_stations.csv"

fig_out = os.path.join(r"/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview", "map_highres_updated.png")


# #FOr comparison
# x = (-104.6,-104.2)
# y = (31.6,31.75)
# fig_out = os.path.join(r"/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview", "map_updated_zoom.jpg")

data = pd.read_csv(output,parse_dates=["origin_time"])
sta_data = pd.read_csv(sta_path)


fig = plt.figure(figsize=(14,8))
# fig = plt.figure(figsize=plt.rcParams["figure.figsize"] * np.array([1.5, 1]))
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

# # Panel C: Latitude vs Depth
# ax2 = fig.add_subplot(grd[1, 1])
# ax2.plot(data["latitude"], data["depth"], 'k.', markersize=2, alpha=1.0, rasterized=True)
# ax2.set_xlim(np.array(y))
# ax2.set_ylim([0, 12])
# ax2.invert_yaxis()
# ax2.set_xlabel("Latitude (°)")
# ax2.set_ylabel("Depth (km)")
# ax2.set_prop_cycle(None)
# ax2.plot(y[0] - 10, 31, '.', markersize=10)


#panel C
x = (-104.84329,-103.79942)
y = (31.3961,31.91505)
starttime = UTCDateTime(2017, 1, 1)
endtime   = UTCDateTime(2024, 8, 1)
output = "/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview/aoi.csv"


data = pd.read_csv(output,parse_dates=["origin_time"])
print(data)
data["m"] = 2*(1.1**(np.array(data["magnitude"])))
ax2 = fig.add_subplot(grd[1, 1])
ax2.hist(data["origin_time"], range=(starttime.datetime, endtime.datetime), 
         bins = mdates.drange(starttime.datetime, endtime.datetime, 
                              pd.Timedelta(weeks=4)), 
         color = "k", edgecolor = "w", alpha = 0.85, linewidth = 0.5,
         label = f"Events = {len(data['magnitude'])}"
         )
ax2.set_ylabel("Count", 
            #    size=18,
               color="black")
ax2.set_xlabel("Date", 
            #    size=18,
               color="red")
# ax.set_title("Number of events", size=18, weight='bold')  # Fixed title syntax

ax2.autoscale(enable=True, axis='x', tight=True)  # Use ax instead of fig.gca()
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.grid(True, which="major", linestyle="--", alpha=0.7,color="red")
ax2.tick_params(axis='x', colors='red',
                # labelsize=18
                )
ax2.tick_params(axis='y', colors='black',
                # labelsize=18
                )
ax2.grid(False, axis="y")

fig.autofmt_xdate()  # Corrected this line
ax2.spines["bottom"].set_edgecolor('red')
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.legend(loc='upper left', fontsize=12)

ax2_r = ax2.twinx()
norm = mpl.colors.Normalize(vmin=0, vmax=15)
ev = ax2_r.scatter(data["origin_time"], data["magnitude"],
                   s=data["m"], c='green', edgecolor=None,
                   alpha=0.3)
#cb = fig.colorbar(ev, orientation='vertical')
#cb.ax.set_ylabel('Profundidad (km)', size=20)
#cb.ax.tick_params(labelsize=20)
ax2_r.set_ylabel('Mw', 
                #  size=20,
                 color="green")
ax2_r.spines["right"].set_edgecolor('green')
ax2_r.tick_params(axis='y', colors='green')
# ax2_r.yaxis.offsetText.set_fontsize(20)
ax2_r.set_ylim(1,6)
ax2_r.tick_params(labelsize=15)

# Store all axes in a list (order matters!)
axes = [ax0, ax1, ax2]

# Auto-labeling with letters (a), (b), (c), ...
for n, ax in enumerate(axes,3):
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