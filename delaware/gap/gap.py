#                           ISOGAP 0.1
#This script calculates the primary azimuthal gap for a certain network geometry
#Input:
#- A csv file with station name, longitude and latitude columns
#- Corners of a rectangle to create the grided area
#- Grid step
## Author: Nelson David Perez e-mail:ndperezg@gmail.com 

import os
import pandas as pd
import numpy as np
import nvector as nv
import sys
from itertools import combinations
import concurrent.futures as cf

def gaps(azimuths):
    """
    Calculate the gaps between sorted azimuth values.

    Args:
        azimuths (list of float): List of azimuth angles.

    Returns:
        list of float: Gaps between successive azimuth values.
    """
    sorted_azimuths = sorted(azimuths)
    gaps_ = []
    
    for i in range(len(sorted_azimuths)):
        if i != 0:
            alpha = sorted_azimuths[i] - sorted_azimuths[i - 1]
        else:
            alpha = sorted_azimuths[0] + 360 - sorted_azimuths[-1]
        gaps_.append(alpha)
    
    return gaps_

def azimuth(lon1, lat1, lon2, lat2):
    """
    Calculate the azimuth angle between two geographic points.

    Args:
        lon1 (float): Longitude of the first point.
        lat1 (float): Latitude of the first point.
        lon2 (float): Longitude of the second point.
        lat2 (float): Latitude of the second point.

    Returns:
        float: Azimuth angle in degrees.
    """
    wgs84 = nv.FrameE(name='WGS84')
    point_a = wgs84.GeoPoint(latitude=lat1, longitude=lon1, z=0, degrees=True)
    point_b = wgs84.GeoPoint(latitude=lat2, longitude=lon2, z=0, degrees=True)
    p_ab_n = point_a.delta_to(point_b)
    
    azim = p_ab_n.azimuth_deg
    
    if azim < 0:
        azim += 360
    
    return azim

def each_gap(lon, lat, net):
    """
    Calculate the maximum azimuth gap for a given point relative to a network of stations.

    Args:
        lon (float): Longitude of the reference point.
        lat (float): Latitude of the reference point.
        net (dict): Dictionary with station names as keys and (longitude, latitude) as values.

    Returns:
        float: Maximum azimuth gap.
    """
    azimuths = []
    
    for station in net:
        print(lon, lat, net[station][0], net[station][1])
        azim = azimuth(lon, lat, net[station][0], net[station][1])
        azimuths.append(azim)
    
    max_gap = max(gaps(azimuths))
    
    return max_gap

def export_gap(net, minlon, maxlon, minlat, maxlat, step, out_path):
    """
    Export the calculated gaps to a CSV file for a grid of longitudes and latitudes.

    Args:
        net (dict): Dictionary with station names as keys and (longitude, latitude) as values.
        minlon (float): Minimum longitude for the grid.
        maxlon (float): Maximum longitude for the grid.
        minlat (float): Minimum latitude for the grid.
        maxlat (float): Maximum latitude for the grid.
        step (float): Step size for longitude and latitude grid.
        out_path (str): Path to the output CSV file.
    """
    if not os.path.isdir(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    lons = np.arange(minlon, maxlon, step)
    lats = np.arange(minlat, maxlat, step)

    with open(out_path, 'w') as out_file:
        out_file.write('longitude,latitude,gap\n')

        for lon in lons:
            for lat in lats:
                az_gap = each_gap(lon, lat, net)
                print(lon, lat, az_gap)
                out_file.write(f'{lon:.5f},{lat:.5f},{az_gap:.2f}\n')

def run_gap(sta_csv, out_folder, grid, step):
    """
    Run the gap calculation and export the results.

    Args:
        sta_csv (str): Path to the CSV file with station data.
        out_folder (str): Path to the folder where results will be saved.
        grid (tuple): Tuple containing (minlon, maxlon, minlat, maxlat).
        step (float): Step size for longitude and latitude grid.
    """
    minlon, maxlon, minlat, maxlat = grid

    df = pd.read_csv(sta_csv)

    net = {}
    for _, row in df.iterrows():
        net[row["station"]] = [row["longitude"], row["latitude"]]

    out_path = os.path.join(out_folder, f"gap_{step}.csv")
    export_gap(net, minlon, maxlon, minlat, maxlat, step, out_path)


if __name__ == "__main__":

    step = 0.01 

    # grid = [-104.84329,-103.79942,31.39610,31.91505] # AOI
    # grid = [-103.9989,-102.9081,30.7996,31.7140] #aux
    grid = [-104.84329,-102.9081,30.7996,31.91505] #total
    # sta_csv = "/home/emmanuel/Ecopetrol/ISOGAP/data/akacias.csv"
    # sta_csv = "/home/emmanuel/Ecopetrol/ISOGAP/data/akacias_2.csv"
    sta_csv = "/home/emmanuel/ecastillo/dev/delaware/data/metadata/delaware_onlystations_160824.csv"
    out_folder = "/home/emmanuel/ecastillo/dev/delaware/data/gap_total"
    run_gap(sta_csv,out_folder,grid,
                step)

