from obspy import UTCDateTime
from delaware.core.client import CustomClient
from delaware.core.eqviewer_utils import get_distance_in_dataframe
import pandas as pd

def prepare_cat2inv(cat, picks, attach_station=None):
    """
    Prepare catalog and picks data for inversion by filtering, cleaning, 
    and computing distances and travel times.
    
    cat and picks are dataframes obtained from delaware.core.client import CustomClient
    
    Parameters:
    - cat (pd.DataFrame): Catalog of events with columns like event ID, origin time, 
                          latitude, longitude, and magnitude.
    - picks (pd.DataFrame): Picks data with columns including event ID, station, 
                            phase hint (P or S), and arrival time.
    - attach_station (pd.DataFrame, optional): Optional DataFrame to attach station 
                                              latitude and longitude.

    Returns:
    - tuple: Updated catalog DataFrame and modified picks DataFrame.
    """
    
    # Rename columns in the catalog to standardize names for latitude and longitude
    cat.columns = cat.columns.get_level_values(1)
    cat = cat.rename(columns={"latitude": "eq_latitude", "longitude": "eq_longitude"})

    # Convert arrival and origin times to datetime for time-based calculations
    picks["arrival_time"] = pd.to_datetime(picks["arrival_time"])
    cat["origin_time"] = pd.to_datetime(cat["origin_time"])

    # Define the essential columns to keep in each DataFrame
    cat_columns = ["ev_id", "origin_time", "eq_latitude", "eq_longitude", "magnitude"]
    picks_columns = ["ev_id", "station", "phase_hint", "arrival_time"]

    # Remove duplicate events in the catalog, keeping the first occurrence
    cat = cat.drop_duplicates(ignore_index=True, keep="first")
    
    # Drop picks with missing phase hints and filter for P and S phase hints only
    picks = picks.dropna(subset="phase_hint")
    picks = picks[picks["phase_hint"].isin(["P", "S"])]
    
    # Remove duplicate picks by event ID, station, and phase hint
    picks = picks.drop_duplicates(subset=["ev_id", "station", "phase_hint"], ignore_index=True)

    # Select only the specified columns from the catalog
    cat = cat[cat_columns]
    
    # If station coordinates are provided, merge them with picks and process further
    if attach_station is not None:
        # Merge station coordinates with picks data
        picks = pd.merge(picks, attach_station, on=["network", "station"], how="inner")
        picks_columns += ["latitude", "longitude"]
        
        # Select only the relevant columns from picks
        picks = picks[picks_columns]
        
        # Merge picks with the catalog data on event ID
        picks = pd.merge(picks, cat, on=["ev_id"])
        
        # Calculate distance and azimuth between event and station locations
        picks = get_distance_in_dataframe(
            data=picks, lat1_name="latitude", lon1_name="longitude", 
            lat2_name="eq_latitude", lon2_name="eq_longitude"
        )

        # Calculate travel time as the difference between arrival and origin time
        picks["tt"] = (picks["arrival_time"] - picks["origin_time"]).dt.total_seconds()

        # Keep only the necessary columns
        picks = picks[picks_columns + ["tt", "r", "az"]]
        
        # Reset the index of the DataFrame
        picks.reset_index(inplace=True, drop=True)
        
        # Pivot the DataFrame to have separate columns for P and S phases
        picks = picks.pivot(index=["ev_id", "station", "r", "az"], 
                            columns="phase_hint", values=["arrival_time", "tt"]).reset_index()

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