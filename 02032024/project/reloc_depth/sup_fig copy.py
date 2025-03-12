import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cross_plot_data = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/basement/cross_plot_data_31.7_-104.8_31.7_-103.8.csv")  # Update with actual file path
cross_elv_data = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/basement/cross_elevation_plot_data_31.7_-104.8_31.7_-103.8.csv")  # Update with actual file path
cross_elv_data["Elevation"] = cross_elv_data["Elevation"]*-1/1e3

# Load your DataFrame (replace this with your actual DataFrame)
df = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/02032024/project/depth/reloc_events.csv")  # Replace with actual loading method if necessary

df_z = df[["ev_id","longitude","latitude","z"]]
df_z["Author"] = "S-P Depth Reloc"
df_z.rename(columns={"z":"depth"}, inplace=True)
df_depth = df[["ev_id","longitude","latitude","depth"]]
df_depth["Author"] = "TexNet HighRes"

df = pd.concat([df_z, df_depth])
df = df.sort_values("Author", ascending=False)

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

df_depth["depth_corrected"] = df_depth["depth"] + np.interp(df_depth["longitude"], cross_elv_data["Longitude"], cross_elv_data["Elevation"])
df_z["depth_corrected"] = df_z["depth"] + np.interp(df_z["longitude"], cross_elv_data["Longitude"], cross_elv_data["Elevation"])

sns.scatterplot(x="longitude", y="depth_corrected", color="gray", 
                data=df_depth, label="TexNet HighRes", ax=ax, s=20, alpha=0.5)
sns.scatterplot(x="longitude", y="depth_corrected", color="darkorange", 
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
