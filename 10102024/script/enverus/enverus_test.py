import lasio
import matplotlib.pyplot as plt

# Load the LAS file
path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/data/42-109-00203-00-00_DIG_RS0032306126.LAS"
las = lasio.read(path)

df = las.df()


# Replace missing values
df.replace(-999.25, float('NaN'), inplace=True)

# Plot a specific log
plt.figure(figsize=(6, 10))
plt.plot(df['GR'], df.index, label="Gamma Ray", color='green')
plt.gca().invert_yaxis()
plt.xlabel("Gamma Ray (API)")
plt.ylabel("Depth (m)")
plt.title("Gamma Ray Log")
plt.legend()
plt.grid()
plt.show()
# print(df)