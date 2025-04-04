import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from project.reloc_depth.utils import latlon2yx_in_km
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/reloc_depth/events_initial.csv"
df = pd.read_csv(data_path,parse_dates=["origin_time"])
df = df.rename(columns={"x[km]":"x","y[km]":"y","depth_TexNet_GrowClust":"z_ori","depth_S-P_ZReloc":"z_new"})
df["Author_new"] = df.apply(lambda x: "Relative Reloc" if pd.isna(x["z_new"]) else "S-P Depth Reloc",axis=1)

df["S-P"] = ~df['z_new'].isna()

# Load DataFrame (assuming df is already loaded with the given structure)
while df['z_new'].isna().any():
    # Identify reference events (those with both z_ori and z_new)
    ref_events = df.dropna(subset=['z_new'])
    target_events = df[df['z_new'].isna()].copy()
    
    
    # Build a KDTree for efficient spatial searching
    ref_tree = cKDTree(ref_events[['x', 'y']].values)
    
    # Power parameter for IDW
    p = 2  # Adjust this if needed
    max_distance = 500 # 1 km radius limit
    epsilon = 1e6
    
    # Function to estimate z_new
    for idx, row in target_events.iterrows():
        dists, idxs = ref_tree.query([row['x'], row['y']], k=5, distance_upper_bound=max_distance)
        valid = dists < max_distance  # Filter out large distances
        
        if np.any(valid):
            dists, idxs = dists[valid], idxs[valid]
            weights = 1 / ((dists ** p) + epsilon)  # Inverse distance weighting
            delta_z = ref_events.iloc[idxs]['z_new'].values - ref_events.iloc[idxs]['z_ori'].values
            target_events.at[idx, 'z_new'] = row['z_ori'] + np.sum(weights * delta_z) / np.sum(weights)
            
        else:
            target_events.at[idx, 'z_new'] = row['z_ori']  # If no neighbors, keep original depth
    
    # Update the original DataFrame
    df.update(target_events[['ev_id', 'z_new', 'Author_new']])


df['Author_ori'] = "TexNet HighRes"
# df_rel = df[["ev_id","longitude","latitude","z_new","S-P","station","region","Author_new"]]
# # df_rel["Author"] = "S-P Depth Reloc"
# df_rel.rename(columns={"z_new":"depth"}, inplace=True)
# df_ori = df[["ev_id","longitude","latitude","z_ori","S-P","station","region","Author_new"]]
# df_ori.rename(columns={"z_ori":"depth"}, inplace=True)

# df = pd.concat([df_rel, df_ori])
# df = df.sort_values("Author_new", ascending=True)
print(df)
df.to_csv("/home/emmanuel/ecastillo/dev/delaware/02032024/project/reloc_depth/reloc_events.csv", index=False)
exit()

# Plot Longitude vs Depth on the first subplot
# Create a figure with two subplots (side by side)
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns


# sns.scatterplot(x="longitude", y="depth", hue="Author", palette=["blue", "red"], data=df[~df["S-P"]], ax=axes[0], marker="+")
# sns.scatterplot(x="longitude", y="depth", hue="Author", palette=["blue", "red"], data=df[df["S-P"]], ax=axes[0])
# sns.kdeplot(x="longitude", y="depth",  data=df_depth, ax=axes[0], color="blue", fill=True, alpha=0.3)
# sns.kdeplot(x="longitude", y="depth",  data=df_z, ax=axes[0], color="red", fill=True, alpha=0.3)

# # Set labels and title for the first subplot
# axes[0].set_xlabel("Longitude", fontsize=12)
# axes[0].set_ylabel("Depth", fontsize=12)
# axes[0].set_ylim(20, 0)  # Invert the y-axis for depth
# axes[0].set_title("Longitude vs Depth", fontsize=14)
# axes[0].legend(loc="lower left")

# # Plot Latitude vs Depth on the second subplot
# sns.scatterplot(x="latitude", y="depth", hue="Author", palette=["blue", "red"], data=df[~df["S-P"]], ax=axes[1], marker="+")
# sns.scatterplot(x="latitude", y="depth", hue="Author", palette=["blue", "red"], data=df[df["S-P"]], ax=axes[1])
# sns.kdeplot(x="latitude", y="depth",  data=df_depth, ax=axes[1], color="blue", fill=True, alpha=0.3)
# sns.kdeplot(x="latitude", y="depth",  data=df_z, ax=axes[1], color="red", fill=True, alpha=0.3)

# # Set labels and title for the second subplot
# axes[1].set_xlabel("Latitude", fontsize=12)
# axes[1].set_ylabel("Depth", fontsize=12)
# axes[1].set_ylim(20, 0)  # Invert the y-axis for depth
# axes[1].set_title("Latitude vs Depth", fontsize=14)
# axes[1].legend(loc="lower right")

# # Adjust layout to avoid overlap
# plt.tight_layout()

# # Show the plot
# plt.show()