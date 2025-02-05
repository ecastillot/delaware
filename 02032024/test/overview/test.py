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

data = pd.read_csv(output,parse_dates=["origin_time"])

# Plot histogram with bins centered on each year
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(data['yr'], bins=range(min(data['yr']), max(data['yr']) + 1),
        color="green",
        edgecolor='black', alpha=0.7)

# Adjust x-axis ticks to be centered on the year
ax.set_xticks(range(min(data['yr']), max(data['yr']) + 1))

# Set labels and title
ax.set_ylabel("Count", size=18,color="black")
ax.set_xlabel("Date", size=18,color="black")
ax.set_title('Histogram of Events by Year', size=18, weight='bold') 
ax.grid(True, which="major", linestyle="--", alpha=0.7,color="gray")
ax.tick_params(axis='x', colors='black',labelsize=18)
ax.tick_params(axis='y', colors='black',labelsize=18)

# Show the plot
fig.savefig(os.path.join(r"/home/emmanuel/ecastillo/dev/delaware/02032024/test/overview", "hist_year.jpg"), 
            bbox_inches = "tight", dpi = 300)
plt.show()