import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load your DataFrame (replace this with your actual DataFrame)
df = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/02032024/project/depth/reloc_events.csv")  # Replace with actual loading method if necessary

df_z = df[["ev_id","longitude","latitude","z"]]
df_z["Author"] = "S-P Depth Reloc"
df_z.rename(columns={"z":"depth"}, inplace=True)
df_depth = df[["ev_id","longitude","latitude","depth"]]
df_depth["Author"] = "TexNet HighRes"

df = pd.concat([df_z, df_depth])
df = df.sort_values("Author", ascending=False)

# Create a figure with two subplots (side by side)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

# Plot Longitude vs Depth on the first subplot
sns.scatterplot(x="longitude", y="depth", hue="Author", palette=["blue", "red"], data=df, ax=axes[0],
                # marker="+",
                s=20)
sns.kdeplot(x="longitude", y="depth",  data=df_depth, ax=axes[0], color="blue", fill=True, alpha=0.3)
sns.kdeplot(x="longitude", y="depth",  data=df_z, ax=axes[0], color="red", fill=True, alpha=0.3)

# Set labels and title for the first subplot
axes[0].set_xlabel("Longitude", fontsize=12)
axes[0].set_ylabel("Depth", fontsize=12)
axes[0].set_xlim(-104.8,-103.8)
axes[0].set_ylim(20, 0)  # Invert the y-axis for depth
axes[0].set_title("Longitude vs Depth", fontsize=14)
axes[0].legend(loc="lower left")

# Plot Latitude vs Depth on the second subplot
sns.scatterplot(x="latitude", y="depth", hue="Author", 
                palette=["blue", "red"], data=df, ax=axes[1],
                # marker="+",
                s=20)
sns.kdeplot(x="latitude", y="depth",  data=df_depth, ax=axes[1], color="blue", fill=True, alpha=0.3)
sns.kdeplot(x="latitude", y="depth",  data=df_z, ax=axes[1], color="red", fill=True, alpha=0.3)

# Set labels and title for the second subplot
axes[1].set_xlim(31.575,31.750)
axes[1].set_xlabel("Latitude", fontsize=12)
axes[1].set_ylabel("Depth", fontsize=12)
axes[1].set_ylim(20, 0)  # Invert the y-axis for depth
axes[1].set_title("Latitude vs Depth", fontsize=14)
axes[1].legend(loc="lower right")

# Adjust layout to avoid overlap
plt.tight_layout()


fig.savefig("/home/emmanuel/ecastillo/dev/delaware/02032024/project/depth/plot_z.png")
# Show the plot
plt.show()