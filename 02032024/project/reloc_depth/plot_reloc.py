import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from project.reloc_depth.utils import latlon2yx_in_km
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/reloc_depth/reloc_events.csv"
df = pd.read_csv(data_path)

df_depth = df[df["Author"] == "TexNet HighRes"]
df_z = df[df["Author"] == "S-P Depth"]


# Plot Longitude vs Depth on the first subplot
# Create a figure with two subplots (side by side)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns


sns.scatterplot(x="longitude", y="depth", hue="Author", palette=["blue", "red"], data=df[~df["S-P"]], ax=axes[0], marker="+")
sns.scatterplot(x="longitude", y="depth", hue="Author", palette=["blue", "red"], data=df[df["S-P"]], ax=axes[0])
sns.kdeplot(x="longitude", y="depth",  data=df_depth, ax=axes[0], color="blue", fill=True, alpha=0.3)
sns.kdeplot(x="longitude", y="depth",  data=df_z, ax=axes[0], color="red", fill=True, alpha=0.3)

# Set labels and title for the first subplot
axes[0].set_xlabel("Longitude", fontsize=12)
axes[0].set_ylabel("Depth", fontsize=12)
axes[0].set_ylim(20, 0)  # Invert the y-axis for depth
axes[0].set_title("Longitude vs Depth", fontsize=14)
axes[0].legend(loc="lower left")

# Plot Latitude vs Depth on the second subplot
sns.scatterplot(x="latitude", y="depth", hue="Author", palette=["blue", "red"], data=df[~df["S-P"]], ax=axes[1], marker="+")
sns.scatterplot(x="latitude", y="depth", hue="Author", palette=["blue", "red"], data=df[df["S-P"]], ax=axes[1])
sns.kdeplot(x="latitude", y="depth",  data=df_depth, ax=axes[1], color="blue", fill=True, alpha=0.3)
sns.kdeplot(x="latitude", y="depth",  data=df_z, ax=axes[1], color="red", fill=True, alpha=0.3)

# Set labels and title for the second subplot
axes[1].set_xlabel("Latitude", fontsize=12)
axes[1].set_ylabel("Depth", fontsize=12)
axes[1].set_ylim(20, 0)  # Invert the y-axis for depth
axes[1].set_title("Latitude vs Depth", fontsize=14)
axes[1].legend(loc="lower right")

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()