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
# Set x-axis limits for specific months
# Dates
x = (-104.84329,-103.79942)
y = (31.3961,31.91505)
starttime = UTCDateTime(2017, 1, 1)
endtime   = UTCDateTime(2024, 6, 1)
start_zoom = datetime(2017, 5, 1)
end_zoom = datetime(2024, 5, 31)
events_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust/origin.csv"
output = "/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview/aoi.csv"
# events = get_texnet_high_resolution_catalog(events_path,xy_epsg="EPSG:3116",
#                                             author="texnet")


# events.filter_rectangular_region(x+y)
# data = events.data

# data.to_csv(output,index=False)


data = pd.read_csv(output,parse_dates=["origin_time"])
print(len(data))
print(data["origin_time"].min())
print(data["origin_time"].max())
data = data[data["magnitude"]>=2]

print(len(data))
print(data["origin_time"].min())
print(data["origin_time"].max())

# starttime = data["origin_time"].min().replace(day=1,hour=0,minute=0,
#                                               second=0,microsecond=0)
# print(mdates.drange(starttime.datetime, endtime.datetime, pd.Timedelta(months=1)))

# exit()
data["m"] = 2*(2**(np.array(data["magnitude"])))

# print(events.__str__(True))
bins = pd.date_range(start=starttime.datetime, end=endtime.datetime, freq='MS')
fig, ax = plt.subplots(figsize=(15, 10))
ax.hist(data["origin_time"], range=(starttime.datetime, endtime.datetime), 
         bins = bins, 
         color = "k", edgecolor = "w", alpha = 0.85, linewidth = 0.5,
         label = f"Events = {len(data['magnitude'])}")
ax.set_ylabel("Count", size=18,color="black")
ax.set_xlabel("Date", size=18,color="black")
# ax.set_title("Number of events", size=18, weight='bold')  # Fixed title syntax

ax.autoscale(enable=True, axis='x', tight=True)  # Use ax instead of fig.gca()

# Major ticks: one per year
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.tick_params(axis='x', which='major', length=10, width=2)  # For yearly ticks

# Minor ticks: one per month (no label, but for grid)
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
ax.grid(True, which='major', linestyle='--', linewidth=1.5)

ax.set_xlim(start_zoom, end_zoom)

ax.tick_params(axis='x', colors='black',labelsize=18)
ax.tick_params(axis='y', colors='black',labelsize=18)
ax.grid(False, axis="y")

fig.autofmt_xdate()  # Corrected this line
ax.spines["bottom"].set_edgecolor('black')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax.legend(loc='upper left', fontsize=18)

ax2 = ax.twinx()
norm = mpl.colors.Normalize(vmin=0, vmax=15)
ev = ax2.scatter(data["origin_time"], data["magnitude"], 
                 s=data["m"], c='darkorange', edgecolor=None, alpha=0.6)
#cb = fig.colorbar(ev, orientation='vertical')
#cb.ax.set_ylabel('Profundidad (km)', size=20)
#cb.ax.tick_params(labelsize=20)
ax2.set_ylabel('Magnitude', size=20,color="darkorange")
ax2.spines["right"].set_edgecolor('darkorange')
ax2.spines["right"].set_linewidth(2)
ax2.tick_params(axis='y', colors='darkorange')
ax2.yaxis.offsetText.set_fontsize(20)
ax2.set_ylim(2,6)
ax2.tick_params(labelsize=15)
ax2.tick_params(
    axis='y',        # y-axis
    which='major',   # major ticks (you can also do 'minor')
    width=4,       # make ticks thicker
    length=10,       # make ticks longer
    color='darkorange',  # tick color
    labelsize=15     # tick label font size
)


fig.savefig(os.path.join(r"/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview", "hist.png"), bbox_inches = "tight", dpi = 300)
plt.show()