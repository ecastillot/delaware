import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

edge = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/eq/edges.csv"
edge = pd.read_csv(edge)

# plt.figure(figsize=(12, 6))
plt.figure()
plt.subplot(1, 1,1)
sns.histplot(edge, x='w', hue='real_link', kde=True, stat="density", common_norm=False)
plt.title('Distribution of w for True and False in real_link')
plt.show()