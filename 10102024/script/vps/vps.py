from delaware.core.read import EQPicks
from delaware.core.eqviewer import Stations
from delaware.loc.inv import prepare_cat2vps
import pandas as pd
import os
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

def compute_vij(arrivals):
    # Ensure arrival times are in datetime format
    arrivals['arrival_time_P'] = pd.to_datetime(arrivals['arrival_time_P'])
    arrivals['arrival_time_S'] = pd.to_datetime(arrivals['arrival_time_S'])
    
    # List to store results
    results = []

    # Iterate over all combinations of row indices
    for i, j in combinations(arrivals.index, 2):
        # Compute v_ij
        delta_t_S = (arrivals.loc[i, 'arrival_time_S'] - arrivals.loc[j, 'arrival_time_S']).total_seconds()
        delta_t_P = (arrivals.loc[i, 'arrival_time_P'] - arrivals.loc[j, 'arrival_time_P']).total_seconds()
        
        if delta_t_P != 0:  # Avoid division by zero
            v_ij = delta_t_S / delta_t_P
            
            if v_ij > 0:
            #     if v_ij >3:
            #         print(arrivals.loc[i, 'station'],
            #               arrivals.loc[j, 'station'],
            #               delta_t_S,delta_t_P,
            #               v_ij)
            #     else:
                
                results.append({
                    'station_i': arrivals.loc[i, 'station'],
                    'station_j': arrivals.loc[j, 'station'],
                    'v_ij': v_ij
                })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def plot_vij_histogram(vij_df, bins=20):
    """
    Plots a histogram of the v_ij values.

    Parameters:
        vij_df (pd.DataFrame): DataFrame containing v_ij values.
        bins (int): Number of bins for the histogram.
    """
    # Extract v_ij values
    v_ij_values = vij_df['v_ij']
    
    # Plot the histogram using Seaborn
    plt.figure(figsize=(8, 5))
    sns.histplot(v_ij_values, bins=bins, kde=True, 
                 color='skyblue', edgecolor='black')
    
    # Add labels and title
    plt.title("Histogram of $v_{i,j}$ Values", fontsize=16)
    plt.xlabel("$v_{i,j}$", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    
    # Show the plot
    plt.grid(alpha=0.3)
    plt.show()



c1 = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/clusters/C1.bna"
root = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust"
catalog_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/clusters/eq_c1.csv"
picks_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust/picks.db"
author = "growclust"
proj = "EPSG:3857"
stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/standard_stations.csv"

stations = pd.read_csv(stations_path)
stations_columns = ["network","station","latitude","longitude","elevation","x[km]","y[km]"]
stations = stations[stations_columns]
stations["station_index"] = stations.index
stations = Stations(data=stations,xy_epsg=proj)


c1 =pd.read_csv(c1)
c1 = list(zip(c1['lon'], c1['lat']))

eqpicks = EQPicks(root,author=author,
                      xy_epsg=proj,
                      catalog_path=catalog_path,
                      picks_path=picks_path
                      )
stations.filter_general_region(c1)
cat,picks = eqpicks.get_catalog_with_picks(
                                        general_region=c1
                                           )
print(picks)
cat,picks = prepare_cat2vps(cat.data,picks.data,stations.data)
# print(picks)

all_vps = []
picks = picks.groupby(by="ev_id")
for ev_id, arrivals in picks.__iter__():
    vij_results = compute_vij(arrivals)
    # print(vij_results)
    if vij_results.empty:
        continue
    else:
        # vij_results.hist("v_ij",bins=20)
        # plt.show()
        all_vps.append(vij_results)
        
vps = pd.concat(all_vps,ignore_index=True)
print(vps.describe())
plot_vij_histogram(vij_df=vps,bins=10)
vps.hist("v_ij",bins=10)
# plt.plot(x=vps
#     "v_ij"],y=vps["v_ij"])
# plt.show()




# print(vps)
# plot_vij_histogram(vps)
    # arrivals.reset_index()
    # # Generate all possible combinations without repetition
    # combinations_list = list(combinations(arrivals.index, 2))  # Pair of indices (no repeats)
    # print(combinations_list)
    # # Create a new DataFrame with combinations
    # combinations_df = pd.DataFrame([
    #     (arrivals.loc[i], arrivals.loc[j]) for i, j in combinations_list
    # ], columns=['Row1', 'Row2'])

    # # Reset the index for readability
    # combinations_df.reset_index(drop=True, inplace=True)
    # print(combinations_df)
    # print(ev_id,arrivals)
    # print(vij_results)
# print(x)
# print(cat)
# print(picks.data)
