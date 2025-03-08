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

# Create a single plot
fig, ax = plt.subplots(figsize=(7, 6))  # Adjust the size as needed

# Plot Longitude vs Depth on the single axis
sns.scatterplot(x="longitude", y="depth", hue="Author", palette=["blue", "red"], data=df, ax=ax, s=20)
sns.kdeplot(x="longitude", y="depth", data=df_depth, ax=ax, color="blue", fill=True, alpha=0.3)
sns.kdeplot(x="longitude", y="depth", data=df_z, ax=ax, color="red", fill=True, alpha=0.3)

# Set labels and title for the plot
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Depth", fontsize=12)
ax.set_xlim(-104.8, -103.8)
ax.set_ylim(20, 0)  # Invert the y-axis for depth
ax.set_title("Longitude vs Depth", fontsize=14)
ax.legend(loc="lower left")

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the plot
fig.savefig("/home/emmanuel/ecastillo/dev/delaware/02032024/project/reloc_depth/sup_fig_z.png")

# Show the plot
plt.show()
