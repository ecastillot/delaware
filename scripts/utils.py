import os
import pandas as pd
from obspy import read_inventory
from openpyxl import Workbook

def get_main_project_path():
    """
    Return the directory path of the current script.

    This function uses the __file__ attribute to determine the directory 
    of the script in which this function is defined.
    """
    return os.path.dirname(__file__)

def get_inventory_info(inventory):
    """
    Extracts channel information from an inventory object and sorts the channels by start date.

    Args:
        inventory (Inventory): Obspy inventory object

    Returns:
        DataFrame: A dataframe containing channel information sorted by start date.
    """
    channel_info = {
        "network": [],
        "station": [],
        "station_latitude": [],
        "station_longitude": [],
        "station_elevation": [],
        "station_starttime": [],
        "station_endtime": [],
        "channel": [],
        "location_code": [],
        "latitude": [],
        "longitude": [],
        "elevation": [],
        "depth": [],
        "site": [],
        "epoch": [],
        "starttime": [],
        "endtime": [],
        "equipment": [],
        "sampling_rate": [],
        "sensitivity": [],
        "frequency": [],
        "azimuth": [],
        "dip": [],
    }
    
    def get_start_date(channel):
        return channel.start_date
    
    for network in inventory:
        for station in network:


            # Sort the channels based on their start dates
            sorted_channels = sorted(station, key=get_start_date)

            epochs = {}

            for channel in sorted_channels:
                
                # channel_info["network"].append(network.code)
                channel_info["network"].append(network.code)
                channel_info["station"].append(station.code)
                channel_info["station_latitude"].append(station.latitude)
                channel_info["station_longitude"].append(station.longitude)
                channel_info["station_elevation"].append(station.elevation)
                channel_info["station_starttime"].append(station.start_date)
                channel_info["station_endtime"].append(station.end_date)
                channel_info["channel"].append(channel.code)
                channel_info["location_code"].append(channel.location_code)
                channel_info["latitude"].append(channel.latitude)
                channel_info["longitude"].append(channel.longitude)
                channel_info["elevation"].append(channel.elevation)
                channel_info["depth"].append(channel.depth)
                channel_info["site"].append(station.site.name)
                channel_info["starttime"].append(channel.start_date)
                channel_info["endtime"].append(channel.end_date)
                channel_info["sampling_rate"].append(channel.sample_rate)
                
                if channel.sensor is None:
                    channel_info["equipment"].append(None)
                else:
                    channel_info["equipment"].append(channel.sensor.type)
                
                if channel.code not in list(epochs.keys()):
                    epochs[channel.code] = 0
                else:
                    epochs[channel.code] += 1
                
                channel_info["epoch"].append(epochs[channel.code])
                
                instrument_type = channel.code[:2]
                if instrument_type == "HN":
                    output_freq_gain = "ACC"
                else:
                    output_freq_gain = "VEL"
                
                channel.response.recalculate_overall_sensitivity()
                freq,gain = channel.response._get_overall_sensitivity_and_gain(frequency=1.0,output = output_freq_gain)
                channel_info["sensitivity"].append(gain)
                channel_info["frequency"].append(freq)
                
                channel_info["azimuth"].append(channel.azimuth)
                channel_info["dip"].append(channel.dip)

    channel_info = pd.DataFrame.from_dict(channel_info)
    return channel_info
