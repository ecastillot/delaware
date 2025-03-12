import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec

cross_plot_data = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/basement/cross_plot_data_31.7_-104.8_31.7_-103.8.csv")  # Update with actual file path
cross_elv_data = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/basement/cross_elevation_plot_data_31.7_-104.8_31.7_-103.8.csv")  # Update with actual file path
cross_elv_data["Elevation"] = cross_elv_data["Elevation"]*-1/1e3

# Load your DataFrame (replace this with your actual DataFrame)
df = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/02032024/project/depth/reloc_events.csv")  # Replace with actual loading method if necessary

print(df)

df_z = df[["ev_id","longitude","latitude","z","station","region"]]
df_z["Author"] = "S-P Depth Reloc"
df_z.rename(columns={"z":"depth"}, inplace=True)
df_depth = df[["ev_id","longitude","latitude","depth"]]
df_depth["Author"] = "TexNet HighRes"

df_depth["depth_from_surface"] = df_depth["depth"] + np.interp(df_depth["longitude"], cross_elv_data["Longitude"], cross_elv_data["Elevation"])
df_z["depth_from_surface"] = df_z["depth"] + np.interp(df_z["longitude"], cross_elv_data["Longitude"], cross_elv_data["Elevation"])

df = pd.concat([df_z, df_depth])
df = df.sort_values("Author", ascending=False)



fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(1, 10, figure=fig)  # 2 rows, 3 columns
gs.update(wspace = 0.3, hspace = 2)

axes = []
axes.append(fig.add_subplot(gs[0, 0:8]))  
axes.append(fig.add_subplot(gs[0, 8:10], sharey=axes[0]))  

# First axis

sns.kdeplot(x="longitude", y="depth_from_surface",  
            data=df_z[df_z["region"]==1], ax=axes[0],
            color="magenta", fill=True, alpha=0.3)
sns.kdeplot(x="longitude", y="depth_from_surface",  
            data=df_z[(df_z["region"]==2) & (df_z["depth_from_surface"]>4)], ax=axes[0],
            color="blue", fill=True, alpha=0.3)
sns.kdeplot(x="longitude", y="depth_from_surface",  
            data=df_z[df_z["region"]==3], ax=axes[0],
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


sns.scatterplot(x="longitude", y="depth_from_surface", color="gray", 
                data=df_depth, label="TexNet HighRes", ax=axes[0], s=20, alpha=0.5)
sns.scatterplot(x="longitude", y="depth_from_surface", color="darkorange", 
                data=df_z, label="S-P Depth Reloc", ax=axes[0], s=20, alpha=0.2)




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

axes[1].hist(df_depth["depth_from_surface"],
             bins=50, color="gray",
            #  histtype="step",
             alpha=0.5, 
             orientation="horizontal",density=True)
axes[1].hist(df_z["depth_from_surface"], bins=50, color="darkorange",
            #  histtype="step",
             alpha=0.5, 
             orientation="horizontal",density=True)

axes[1].hist(df_z[df_z["region"]==1]["depth_from_surface"], bins=20, color="magenta",
             histtype="step",
             alpha=0.5, label="Region 1",
             orientation="horizontal",density=True)
axes[1].hist(df_z[df_z["region"]==2]["depth_from_surface"], bins=20, color="blue",
             histtype="step",label="Region 2",
             alpha=0.5, 
             orientation="horizontal",density=True)
axes[1].hist(df_z[df_z["region"]==3]["depth_from_surface"], bins=20, color="green",
             histtype="step",label="Region 3",
             alpha=0.5, 
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
axes[0].legend(loc="lower left", fontsize=14,markerscale=2)

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the plot
fig.savefig("/home/emmanuel/ecastillo/dev/delaware/02032024/project/reloc_depth/sup_fig_z.png")

# Show the plot
plt.show()
exit()

# Create a single plot
fig, ax = plt.subplots(figsize=(7, 6))  # Adjust the size as needed

ax.set_yticks(np.arange(0, 21, 2))  # Major ticks every 0.2

print(cross_elv_data)
# Plot Longitude vs Depth on the single axis
ax.plot(cross_elv_data["Longitude"], cross_elv_data["Elevation"] ,
        color="black", linestyle="-",
        label="Elevation", 
        # linewidth=2
        )
# Plot Longitude vs Depth on the single axis
# ax.plot(cross_plot_data["Longitude"], cross_plot_data["Elevation"], 
#         color="black", linestyle="-",
#         label="Basement", 
#         # linewidth=2
#         )

ax.plot(cross_plot_data["Longitude"], cross_elv_data["Elevation"] + cross_plot_data["Elevation"], 
        color="red", linestyle="--",
        label="Basement")

df_depth["depth_from_surface"] = df_depth["depth"] + np.interp(df_depth["longitude"], cross_elv_data["Longitude"], cross_elv_data["Elevation"])
df_z["depth_from_surface"] = df_z["depth"] + np.interp(df_z["longitude"], cross_elv_data["Longitude"], cross_elv_data["Elevation"])

sns.scatterplot(x="longitude", y="depth_from_surface", color="gray", 
                data=df_depth, label="TexNet HighRes", ax=ax, s=20, alpha=0.5)
sns.scatterplot(x="longitude", y="depth_from_surface", color="darkorange", 
                data=df_z, label="S-P Depth Reloc", ax=ax, s=20, alpha=0.2)


# Set labels and title for the plot
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Depth", fontsize=12)
ax.set_xlim(-104.8, -103.8)
ax.set_ylim(20, -2)  # Invert the y-axis for depth
ax.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)  # Major grid
ax.set_title("Longitude vs Depth", fontsize=14)


# Adjust opacity for individual hues
handles, labels = ax.get_legend_handles_labels()  # Get handles and labels from the legend
for handle, label in zip(handles, labels):
    if label == "S-P Depth Reloc":  # Apply higher opacity for the gray points
        handle.set_alpha(1)  # Set full opacity (gray)
    elif label == "TexNet HighRes":  # Apply lower opacity for the orange points
        handle.set_alpha(0.7)  # Set lower opacity (orange)
ax.legend(loc="lower left")

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the plot
fig.savefig("/home/emmanuel/ecastillo/dev/delaware/02032024/project/reloc_depth/sup_fig_z.png")

# Show the plot
plt.show()
