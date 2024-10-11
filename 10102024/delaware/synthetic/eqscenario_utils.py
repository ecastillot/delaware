import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth
from delaware.core.eqviewer import Stations,Source

def plot_eqscenario_phases(phases:pd.DataFrame,
                stations:Stations,
                sort_by_source:Source=None,
                    show:bool = True):
        
    if not sort_by_source:
        lat = stations.data["latitude"].mean()
        lon = stations.data["longitude"].mean()
        sort_by_source = Source(lat,lon,0,xy_epsg=stations.xy_epsg)
    
    stations = stations.sort_data_by_source(sort_by_source)
    stations["sort_by_r_index"] = stations.index 
    
    if phases.empty:
        print("No phases")
        return None
    
    phases = pd.merge(phases,stations[["station_index","sort_by_r","sort_by_r_index"]],on="station_index")
            
    noises = phases[((phases["event_type"]=="general_noise") | \
                    (phases["event_type"]=="noise2station"))]
    earthquakes = phases[(phases["event_type"]=="earthquake")]
    aftershocks = phases[(phases["event_type"]=="aftershock")]
    
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 6))
    
    ax.scatter(noises["window_travel_time"],noises["sort_by_r"],
                        c="gray")
    
    markers = {"earthquake":"o","aftershock":"o"}
    for eq in [earthquakes,aftershocks]:
        
        if eq.empty:
            continue
        
        events_grouped = eq.groupby("event_index")
        colors = plt.cm.viridis(np.linspace(0, 1, events_grouped.ngroups))
        
        for i,(ev_id,eq_phases) in enumerate(events_grouped):
            
            eq_phases = eq_phases.sort_values(by="travel_time",ignore_index=True)
            first_phase_info = eq_phases.iloc[0]
            
            ev_origin = first_phase_info["window_travel_time"] - first_phase_info["travel_time"] 
            ev_type = first_phase_info['event_type']
            
            ev_loc = gps2dist_azimuth(first_phase_info["src_lat"], first_phase_info["src_lon"],
                                        sort_by_source.latitude, sort_by_source.longitude)[0]/1e3
            
            p_phases = len(eq_phases[eq_phases["phase_hint"]=="P"])
            s_phases = len(eq_phases[eq_phases["phase_hint"]=="S"])
            info = {"ev_type":ev_type,
                    "ev_id":int(ev_id),
                    "r_max":round(eq_phases["sort_by_r"].max(),2),
                    "# Phases": len(eq_phases),
                    "# P-Phases":p_phases,
                    "# S-Phases":s_phases
                        }
            print(info)
            ax.scatter(eq_phases["window_travel_time"],eq_phases["sort_by_r"],
                            marker=markers[ev_type],
                        c=colors[i])
            ax.scatter([ev_origin],[ev_loc],marker="o",linewidths=2,
                        edgecolors=colors[i],facecolors="none",s=120,)
            
    ax.invert_yaxis()
    stats = {"events": len(earthquakes.drop_duplicates("event_index")),
                "aftershocks": len(aftershocks.drop_duplicates("event_index")),
                "true_picks" : len(earthquakes)+len(aftershocks),
                "false_picks": len(noises)
                }
    
    ax.text(0, 1.05, stats, horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes)
    
    ax.set_xlabel("Time (s)")
    src_lat = round(sort_by_source.latitude,2)
    src_lon = round(sort_by_source.longitude,2)
    ax.set_ylabel(f"Distance in km from [lon:{src_lon},lat:{src_lat}]")
    
    if show:
        plt.show()
    
    return ax 

def convert_csv_to_pt_path(csv_path):
    """
    Convert a CSV file path to the corresponding PT file path by:
    1. Replacing 'raw' with 'processed' in the directory path.
    2. Changing the file extension from .csv to .pt.

    Args:
    csv_path (str): The original CSV file path.

    Returns:
    str: The corresponding PT file path.
    """
    # Split the path into directory and file name
    dir_path, file_name = os.path.split(csv_path)

    # Replace 'raw' with 'processed' in the directory path
    new_dir_path = dir_path.replace("raw", "processed")

    # Replace the file extension from .csv to .pt
    file_name = os.path.splitext(file_name)[0] + '.pt'

    # Join the new directory path with the new file name
    pt_path = os.path.join(new_dir_path, file_name)

    return pt_path

def calculate_filepath(i, output_folder, file_format='csv'):
    """Calculate the filepath for a given index i and output folder.

    Args:
        i (int): The index used to calculate the filepath.
        output_folder (str): The output folder where the file will be saved.
        file_format (str, optional): The file format extension. Defaults to 'csv'.

    Returns:
        str: The filepath for the given index i, output folder, and file format.
    """
    
    h_lower_bound = (i // 100) * 100
    h_upper_bound = h_lower_bound + 100
    
    th_lower_bound = (i // 1000) * 1000
    th_upper_bound = th_lower_bound + 1000
    
    # Construct the folder path based on the ranges
    folder = os.path.join(output_folder,f"w_{th_lower_bound}_{th_upper_bound}",
                           f"w_{h_lower_bound}_{h_upper_bound}")
    
    # Create the folder if it doesn't exist
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Construct the filepath for the specific index i
    filepath = os.path.join(folder, f"window_{i}.{file_format}")
    return filepath

def generate_weights(num_elements, priority_factor=2):
    """Generate weights for a range of elements with decreasing priority.

    Args:
        num_elements (int): The number of elements in the range.
        priority_factor (int, optional): The factor by which weights decrease. Defaults to 2.

    Returns:
        list: A list of weights for each element in the range.
    """
    weights = [priority_factor * (num_elements - i) for i in range(num_elements)]
    return weights

def get_distance_in_dataframe(data: pd.DataFrame, lat1_name: str, lon1_name: str,
                              lat2_name: str, lon2_name: str):
    """
    Compute distances between two sets of latitude and longitude coordinates in a DataFrame.

    Args:
    - data (pd.DataFrame): Input DataFrame containing the latitude and longitude columns.
    - lat1_name (str): Name of the column containing the first set of latitudes.
    - lon1_name (str): Name of the column containing the first set of longitudes.
    - lat2_name (str): Name of the column containing the second set of latitudes.
    - lon2_name (str): Name of the column containing the second set of longitudes.

    Returns:
    - pd.DataFrame: DataFrame with additional columns 'r', 'az', 'baz' representing distance (in km),
      azimuth (degrees clockwise from north), and back azimuth (degrees clockwise from south), respectively.
    """
    if data.empty:
        return data
    
    data = data.reset_index(drop=True)
    computing_r = lambda x: gps2dist_azimuth(x[lat1_name], x[lon1_name],
                                             x[lat2_name], x[lon2_name])
    r = data.apply(computing_r, axis=1)
    data[["r", "az", "baz"]] = pd.DataFrame(r.tolist(), index=data.index)
    data["r"] = data["r"] / 1e3
    return data

def generate_random_points(latitude, longitude, depth, radius_km, num_points):
    """
    Generate random points around a given point within a specified radius.

    Args:
    - latitude (float): The latitude of the center point.
    - longitude (float): The longitude of the center point.
    - depth (float): The depth of the center point.
    - radius_km (float): The radius in kilometers.
    - num_points (int): The number of random points to generate.

    Returns:
    - list: A list of tuples, each representing a random point as (latitude, longitude, depth).
    """
    # Calculate the minimum and maximum depths based on the center depth and radius.
    min_depth, max_depth = (depth - radius_km / 2, depth + radius_km / 2)
    # If the minimum depth is less than 0, set it to 0 to avoid negative depths.
    if min_depth < 0:
        min_depth = 0
    points = []
    # Generate random points within the specified radius using a polar coordinate approach.
    for _ in range(num_points):
        angle = np.random.uniform(0, 2 * np.pi)
        u = np.random.uniform(0, radius_km)
        # Convert the polar coordinates to geographic coordinates (latitude, longitude).
        new_latitude = latitude + (u * np.cos(angle)) / 111.32
        new_longitude = longitude + (u * np.sin(angle)) / (111.32 * np.cos(latitude))
        # Randomly assign depths within the calculated range.
        depth = np.random.uniform(min_depth, max_depth)
        points.append((new_latitude, new_longitude, depth))
    return points

def get_travel_time_in_window(tt: pd.DataFrame, 
                              min_n_p_phases: int,
                              min_n_s_phases: int,
                              window: float) -> pd.DataFrame:
    """
    Calculate the travel times of seismic phases within a specified window.

    Parameters:
    -----------
    tt : pd.DataFrame
        Dataframe containing seismic phase travel times.
    min_n_p_phases : int
        Number of minimum P phases.
    min_n_s_phases : int
        Number of minimum S phases.
    window : float
        Total window duration.

    Returns:
    -----------
    pd.DataFrame
        Dataframe containing seismic phases with adjusted travel times within the window.
    """
    # Calculate travel time within the window
    tt["window_travel_time"] = tt["travel_time"] - tt["travel_time"].min()
    
    # Select mandatory phases with smallest travel times
    p_phases = tt[tt["phase_hint"] == "P"]
    p_mandatory_phases = p_phases.nsmallest(min_n_p_phases, 'travel_time')
    s_phases = tt[tt["phase_hint"] == "S"]
    s_mandatory_phases = s_phases.nsmallest(min_n_s_phases, 'travel_time')
    
    mandatory_phases = pd.concat([p_mandatory_phases,s_mandatory_phases])
    last_mandatory_tt = mandatory_phases["travel_time"].max()
    
    # Adjust travel times for all phases
    first_phase = window - last_mandatory_tt
    first_phase_location = random.uniform(0, first_phase)
    tt["window_travel_time"] += first_phase_location
    
    # Filter phases to keep only those within the window
    tt = tt[tt["window_travel_time"] < window]
    
    return tt

def generate_random_with_probability(start:int, end:int, 
                                     prob_range:list, weight_range:tuple):
    """
    Generate a random integer within a specified range with a custom probability distribution.

    Args:
        start (int): The lower bound of the range.
        end (int): The upper bound of the range.
        prob_range (range or list): The range of numbers with higher probability.
        weight_range (tuple): A tuple containing two integers representing the weights
                              for numbers inside and outside the prob_range respectively.

    Returns:
        int: A random integer within the specified range with the custom probability distribution.
    """
    numbers = list(range(start, end + 1))

    # Generate probabilities based on the specified ranges
    probabilities = []
    for num in numbers:
        if num in prob_range:
            probabilities.append(weight_range[0])  # Higher weight for numbers in prob_range
        else:
            probabilities.append(weight_range[1])  # Lower weight for numbers outside prob_range
    
    probabilities = np.array(numbers[::-1])**5
    
    # Generate a random number using the defined probabilities
    random_number = random.choices(numbers, weights=probabilities)[0]
    return random_number


# if __name__ == "__main__":
    