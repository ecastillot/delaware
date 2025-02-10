import pandas as pd
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sci
import numpy as np

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
                
                results.append({
                    'station_i': arrivals.loc[i, 'station'],
                    'station_j': arrivals.loc[j, 'station'],
                    'v_ij': v_ij
                })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def plot_vij_histogram_station(vij_df, color="black", bins=20, output=None,
                               ax=None,legend=False,
                               mode=None,median=None,mean=None,
                               max=None):
    """
    Plots a histogram of the v_ij values.

    Parameters:
        vij_df (pd.DataFrame): DataFrame containing v_ij values.
        cluster (str or int): Cluster identifier for the plot title.
        bins (int, optional): Number of bins for the histogram. Default is 20.
        output (str, optional): If provided, saves the figure to this path.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new figure.
    
    Returns:
        ax (matplotlib.axes.Axes): The axis with the plot.
    """
    # Extract v_ij values
    v_ij_values = vij_df['v_ij']
    
    _median = v_ij_values.quantile(0.5)
    _mean = v_ij_values.mean()
    _mode = sci.stats.mode(np.round(v_ij_values, 1), keepdims=True)[0][0]

    # Create figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # v_ij_values /= v_ij_values.max() 
    v_ij_counts, bin_edges = np.histogram(v_ij_values, bins=bins, density=True)
    v_ij_counts /= v_ij_counts.max() 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    max_bin_index = np.argmax(v_ij_counts)  # Index of the max count
    max_bin_center = bin_centers[max_bin_index]  # Center of that bin
    # print(v_ij_counts , bin_edges)
    # print(v_ij_counts.max() )
    # exit()
    
    ax.plot( bin_centers,v_ij_counts,color=color)
    
    # # Plot histogram using Seaborn
    # sns.histplot(v_ij_values, bins=bins, kde=True, 
    #             #  line_kws={'linewidth': 3},
    #             #  facecolor='none',
    #             # stat="density",
    #             element="poly",
    #              color='none', 
    #              edgecolor=color,
    #              ax=ax)  # Pass ax to Seaborn

    # Add vertical lines for statistics
    if median is not None:
        ax.axvline(x=_median, color=median, linestyle='dashed', label="median")
    if mean is not None:
        ax.axvline(x=_mean, color=mean, linestyle='dashed', label="mean")
    if mode is not None:
        ax.axvline(x=_mode, color=mode , linestyle='dashed', label="mode")
    if max is not None:
        ax.axvline(x=max_bin_center, color=max, linestyle='dashed', label="max")

    # Add labels and title
    # ax.set_title(f"R{cluster}", fontsize=16)
    ax.set_xlabel(r"${v_p}/{v_s}$", fontsize=14)
    ax.set_ylabel("Norm. Counts", fontsize=14)

    if legend:
        ax.legend(loc="upper right")
    # ax.grid(alpha=0.3)
    
    # Set major and minor ticks
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))   # Major ticks every 0.1
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.025)) # Minor ticks every 0.025

    # Grid styling
    ax.grid(True, which='major', linestyle='--', linewidth=0.9, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

    ax.set_xticks(np.arange(1.4, 2, 0.1)) 

    ax.set_xlim(1.4,2)

    # Save plot if output path is provided
    if output is not None:
        plt.savefig(output, dpi=300, bbox_inches='tight')

    return ax

def plot_vij_histogram(vij_df, cluster=None, bins=20, 
                       output=None, ax=None,legend=False,
                       mode=None,median=None,mean=None,
                               max=None):
    """
    Plots a histogram of the v_ij values.

    Parameters:
        vij_df (pd.DataFrame): DataFrame containing v_ij values.
        cluster (str or int): Cluster identifier for the plot title.
        bins (int, optional): Number of bins for the histogram. Default is 20.
        output (str, optional): If provided, saves the figure to this path.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new figure.
    
    Returns:
        ax (matplotlib.axes.Axes): The axis with the plot.
    """
    # Extract v_ij values
    v_ij_values = vij_df['v_ij']
    _median = v_ij_values.quantile(0.5)
    _mean = v_ij_values.mean()
    _mode = sci.stats.mode(np.round(v_ij_values, 1), keepdims=True)[0][0]

    # Create figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    v_ij_counts, bin_edges = np.histogram(v_ij_values, bins=bins, density=True)
    v_ij_counts /= v_ij_counts.max() 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    max_bin_index = np.argmax(v_ij_counts)  # Index of the max count
    max_bin_center = bin_centers[max_bin_index]  # Center of that bin

    # Plot histogram using Seaborn
    sns.histplot(v_ij_values, bins=bins, kde=True, 
                #  stat="density",
                 line_kws={'linewidth': 3},
                 facecolor='none',
                 color='lightcoral', edgecolor='lightcoral',
                 ax=ax)  # Pass ax to Seaborn

    # Add vertical lines for statistics
    if median is not None:
        ax.axvline(x=_median, color=median, linestyle='dashed', label="median")
    if mean is not None:
        ax.axvline(x=_mean, color=mean, linestyle='dashed', label="mean")
    if mode is not None:
        ax.axvline(x=_mode, color=mode , linestyle='dashed', label="mode")
    if max is not None:
        ax.axvline(x=max_bin_center, color=max, linestyle='dashed', label="max")

    # Add labels and title
    # ax.set_title(f"R{cluster}", fontsize=16)
    ax.set_xlabel(r"${v_p}/{v_s}$", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)

    if legend:
        ax.legend(loc="upper right")
    # ax.grid(alpha=0.3)
    
    # Set major and minor ticks
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))   # Major ticks every 0.1
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.025)) # Minor ticks every 0.025

    # Grid styling
    ax.grid(True, which='major', linestyle='--', linewidth=0.9, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

    ax.set_xticks(np.arange(1.4, 2, 0.1)) 
    ax.set_xlim(1.4,1.9)

    # Save plot if output path is provided
    if output is not None:
        plt.savefig(output, dpi=300, bbox_inches='tight')

    return ax