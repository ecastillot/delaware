import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from delaware.core.event.utils import get_distance_in_dataframe

def prepare_sp_analysis(cat, picks, cat_columns_level=1):
    """
    Prepare catalog and picks data for inversion by filtering, cleaning, 
    and computing distances and travel times.
    
    cat and picks are dataframes obtained from delaware.core.client import CustomClient
    
    Parameters:
    - cat (pd.DataFrame): Catalog of events with columns like event ID, origin time, 
                          latitude, longitude, and magnitude.
    - picks (pd.DataFrame): Picks data with columns including event ID, station, 
                            phase hint (P or S), and arrival time.
    - cata_columns_level (int): to identify the level of the column names
    - attach_station (pd.DataFrame, optional): Optional DataFrame to attach station 
                                              latitude and longitude.

    Returns:
    - tuple: Updated catalog DataFrame and modified picks DataFrame.
    """
    
    # Rename columns in the catalog to standardize names for latitude and longitude
    cat.columns = cat.columns.get_level_values(cat_columns_level)
    cat = cat.rename(columns={"latitude": "eq_latitude", "longitude": "eq_longitude"})

    # Convert arrival and origin times to datetime for time-based calculations
    picks["time"] = pd.to_datetime(picks["time"],format="mixed")
    cat["origin_time"] = pd.to_datetime(cat["origin_time"],format="mixed")
    

    # Define the essential columns to keep in each DataFrame
    cat_columns = ["ev_id", "origin_time", "eq_latitude", "eq_longitude", "magnitude"]
    picks_columns = ["ev_id", "station", "phase_hint", "time","latitude", "longitude"]

    # Remove duplicate events in the catalog, keeping the first occurrence
    cat = cat.drop_duplicates(ignore_index=True, keep="first")
    
    # Drop picks with missing phase hints and filter for P and S phase hints only
    picks = picks.dropna(subset="phase_hint")
    picks = picks[picks["phase_hint"].isin(["P", "S"])]
    
    # Remove duplicate picks by event ID, station, and phase hint
    picks = picks.drop_duplicates(subset=["ev_id", "station", "phase_hint"], ignore_index=True)

    # Select only the specified columns from the catalog
    cat = cat[cat_columns]
    picks = picks[picks_columns]
    
        
    # # Merge picks with the catalog data on event ID
    picks = pd.merge(picks, cat, on=["ev_id"])
    
    # Calculate distance and azimuth between event and station locations
    picks = get_distance_in_dataframe(
        data=picks, lat1_name="eq_latitude", lon1_name="eq_longitude", 
        lat2_name="latitude", lon2_name="longitude"
    )

    # Calculate travel time as the difference between arrival and origin time
    picks["tt"] = (picks["time"] - picks["origin_time"]).dt.total_seconds()

    # Keep only the necessary columns
    picks = picks[picks_columns + ["tt", "r", "az"]]
    
    picks.drop_duplicates(subset=["ev_id", "station", "r", "az","time"],inplace=True)
    
    # Reset the index of the DataFrame
    picks.reset_index(inplace=True, drop=True)
    
    print(picks)
    
    # Pivot the DataFrame to have separate columns for P and S phases
    picks = picks.pivot(index=["ev_id", "station", "r", "az"], 
                        columns="phase_hint", values=["time", "tt"]).reset_index()
    
    # Flatten multi-level columns for simplicity
    picks.columns = ['_'.join(col).strip('_') for col in picks.columns.values]
    
    # Drop rows where either P or S travel time is missing
    picks.dropna(subset=["tt_P", "tt_S"], inplace=True)
    
    # Sort picks by P-phase travel time
    picks.sort_values(by="tt_P", inplace=True, ignore_index=True)
    
    # Select the final set of columns for the picks DataFrame
    picks_columns = ["ev_id", "station", "r", "az", "tt_P", "tt_S"]
    picks = picks[picks_columns]
    
    return cat, picks

def plot_times_by_station(data,title=None,show: bool = True, 
                          ylim=None,
                          savefig:str=None,
                          **plot_kwargs):   
    
    data = data.drop_duplicates("ev_id")
    # grouped = data.groupby('station')['ev_id']
    fig,ax = plt.subplots(1,1,figsize=(10, 6))
    sns.boxplot(data=data,x="station",y="ts-tp",ax=ax,**plot_kwargs)
    
    # Calculate the number of samples for each station
    sample_counts = data['station'].value_counts()
    if "order" in plot_kwargs.keys():
        sample_counts = sample_counts.reindex(plot_kwargs["order"])

    # sample_counts.dropna(inplace=True,ignore_index=True)
    print(sample_counts)

    # Annotate the box plot with sample counts
    for i,(station, count) in enumerate(sample_counts.items()):
        
        if not pd.notna(count):
            print(pd.notna(count))
            continue
        
        # Get the x position of each station
        # x_pos = sample_counts.unique().tolist().index(station)
        # Set the annotation above each box plot
        if ylim is not None:
            posy = ylim[-1]
        else:
            posy = data['ts-tp'].max()
        
        ax.text(i, posy-posy*0.07,
                f'n={int(count)}', 
                ha='center', va='bottom', color='red',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7,),
                fontsize=12)
    if title is not None:
        title = r'$t_{\mathrm{s}} - t_{\mathrm{p}}$' +f' Analysis\n{title}'
        ax.set_title(title,
                        fontdict={"size":14})
    ax.set_xlabel('Stations',fontdict={"size":16})
    ax.set_ylabel(r'$t_{\mathrm{s}} - t_{\mathrm{p}}$ (s)',fontdict={"size":16})
    
    if ylim is not None:
        ax.set_ylim(*ylim)
    
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    
    plt.xticks(fontsize=14)  # Rotate x labels for readability
    plt.yticks(fontsize=14)  # Rotate x labels for readability
    plt.tight_layout()  # Adjust layout to avoid cutting off labels
        
    if savefig is not None:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig,ax