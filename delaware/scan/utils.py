# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-08-17 11:34:00
#  * @modify date 2024-08-17 11:34:00
#  * @desc [description]
#  */
import datetime as dt
import numpy as np
import pandas as pd

def process_stream_common_channels(st, location_preferences, instrument_preferences):
    """
    Process the common channels information from an ObsPy Stream object and filter based on preferences.

    Args:
        st (obspy.Stream): The ObsPy Stream object to process.
        location_preferences (list): List of preferred locations to filter by.
        instrument_preferences (list): List of preferred instruments to filter by.

    Returns:
        obspy.Stream: A filtered ObsPy Stream object based on the provided preferences.
    """
    # Extract common channels information as a list of dictionaries
    list_of_dicts = [
        {
            'network': v[0],
            'station': v[1],
            'location': v[2],
            'instrument': v[3][0:2]
        }
        for v in st._get_common_channels_info().keys()
    ]

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(list_of_dicts)

    # Filter the DataFrame by location preferences
    df = filter_by_preference(df, preferences=location_preferences, column="location")

    # Filter the DataFrame by instrument preferences
    df = filter_by_preference(df, preferences=instrument_preferences, column="instrument")

    # Convert the filtered DataFrame to a dictionary for selection
    sel = df.to_dict()

    # Select the appropriate stream data based on the filtered preferences
    st = st.select(
        network=sel["network"][0],
        station=sel["station"][0],
        location=sel["location"][0],
        channel=sel["instrument"][0] + "?"
    )

    return st

def inside_the_polygon(p, pol_points):
    """
    Determine if a point is inside a polygon.

    Parameters:
    -----------
    p : tuple
        Coordinates of the point to check (lon, lat).
    pol_points : list of tuples
        List of coordinates defining the polygon (lon, lat).

    Returns:
    --------
    bool
        True if the point is inside the polygon, False otherwise.
    """
    # Convert list of polygon points to a tuple and close the polygon
    V = tuple(pol_points[:]) + (pol_points[0],)
    cn = 0  # Counter for the number of times the point crosses the polygon boundary

    for i in range(len(V) - 1):
        # Check if the y-coordinate of the point is between the y-coordinates of the edge
        if ((V[i][1] <= p[1] and V[i + 1][1] > p[1]) or
            (V[i][1] > p[1] and V[i + 1][1] <= p[1])):
            # Calculate the intersection point on the x-axis
            vt = (p[1] - V[i][1]) / float(V[i + 1][1] - V[i][1])
            # Check if the point is to the left of the edge
            if p[0] < V[i][0] + vt * (V[i + 1][0] - V[i][0]):
                cn += 1  # Increment the counter for crossing

    # A point is inside the polygon if the number of crossings is odd
    return cn % 2 == 1

def filter_by_preference(df, preferences, column):
    """
    Filter the DataFrame based on a list of preferred values for a specific column.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to filter.
    preferences : list
        List of preferred values to keep in the specified column.
    column : str
        The column name to filter on.

    Returns:
    --------
    pd.DataFrame
        The filtered DataFrame with rows that match the preferred value.
    """
    locations = df[column].to_list()
    loc_pref = None

    for loc_pref in preferences:
        if loc_pref in locations:
            break
    else:
        loc_pref = None

    if loc_pref is not None:
        df = df[df[column] == loc_pref]

    return df

def filter_info(df, remove_networks=None, remove_stations=None,
                location_pref=["", "00", "20", "10", "40"],
                instrument_pref=["HH", "BH", "EH", "HN", "HL"],
                handle_preference="sort",
                domain=[-180, 180, -90, 90]):
    """
    Filter and sort a DataFrame based on multiple criteria.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to filter and sort.
    remove_networks : list of str, optional
        List of network names to remove from the DataFrame. Default is None.
    remove_stations : list of str, optional
        List of station names to remove from the DataFrame. Default is None.
    location_pref : list of str, optional
        List of location codes to use as preferences. Default is ["", "00", "20", "10", "40"].
    instrument_pref : list of str, optional
        List of instrument types to use as preferences. Default is ["HH", "BH", "EH", "HN", "HL"].
    handle_preference : str, optional
        Method for handling preferences, either "remove" to filter by preferences or "sort" to sort by preferences.
        Default is "sort".
    domain : list of float, optional
        List defining the bounding box for filtering, given as [lonw, lone, lats, latn]. Default is [-180, 180, -90, 90].

    Returns:
    --------
    pd.DataFrame
        The filtered and/or sorted DataFrame.
    """
    # Set default values for optional parameters if not provided
    if remove_networks is None:
        remove_networks = []
    if remove_stations is None:
        remove_stations = []

    # Remove rows with specified networks or stations
    df = df[~df["network"].isin(remove_networks)]
    df = df[~df["station"].isin(remove_stations)]

    # Define the polygon for filtering based on domain
    polygon = [
        (domain[0], domain[2]),
        (domain[0], domain[3]),
        (domain[1], domain[3]),
        (domain[1], domain[2]),
        (domain[0], domain[2])
    ]

    # Filter rows based on whether they fall within the specified domain
    if polygon != [(-180, -90), (180, 90)]:
        if polygon[0] != polygon[-1]:
            raise ValueError("The first point must be equal to the last point in the polygon.")
        
        is_in_polygon = lambda x: inside_the_polygon((x.longitude, x.latitude), polygon)
        mask = df[["longitude", "latitude"]].apply(is_in_polygon, axis=1)
        df = df[mask]

    # Add 'instrument' column if not already present
    if "instrument" not in df.columns:
        df["instrument"] = df["channel"].apply(lambda x: x[0:2])
        
    if handle_preference == "remove":
        # Filter by location preferences
        df = filter_by_preference(df, location_pref, "location_code")
        # Filter by instrument preferences
        df = filter_by_preference(df, instrument_pref, "instrument")
        
    elif handle_preference == "sort":
        # Create a mapping for location and instrument preferences
        location_priority = {loc: i for i, loc in enumerate(location_pref)}
        instrument_priority = {instr: i for i, instr in enumerate(instrument_pref)}
        
        # Add priority columns to DataFrame based on preferences
        df['location_priority'] = df['location_code'].map(location_priority)
        df['instrument_priority'] = df['instrument'].map(instrument_priority)
        
        # Sort by the priority columns
        df = df.sort_values(by=['location_priority', 'instrument_priority'])
        
        # Drop the priority columns
        df = df.drop(columns=['location_priority', 'instrument_priority'])

    return df

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
        "instrument": [],
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
                channel_info["instrument"].append(channel.code[0:2])
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
                if not channel.response.response_stages:
                    freq,gain = np.nan,np.nan
                else:
                    channel.response.recalculate_overall_sensitivity()
                    freq,gain = channel.response._get_overall_sensitivity_and_gain(frequency=1.0,output = output_freq_gain)
                
                channel_info["sensitivity"].append(gain)
                channel_info["frequency"].append(freq)
               
                channel_info["azimuth"].append(channel.azimuth)
                channel_info["dip"].append(channel.dip)

    channel_info = pd.DataFrame.from_dict(channel_info)
    return channel_info

def get_chunktimes(starttime,endtime,chunklength_in_sec, 
                   overlap_in_sec=0):
	"""
	Make a list that contains the chunktimes according to 
	chunklength_in_sec and overlap_in_sec parameters.

	Parameters:
	-----------
	starttime: obspy.UTCDateTime object
		Start time
	endtime: obspy.UTCDateTime object
		End time
	chunklength_in_sec: None or int
		The length of one chunk in seconds. 
		The time between starttime and endtime will be divided 
		into segments of chunklength_in_sec seconds.
	overlap_in_sec: None or int
		For more than one chunk, each segment will have overlapping seconds

	Returns:
	--------
	times: list
		List of tuples, each tuple has startime and endtime of one chunk.
	"""

	if chunklength_in_sec == 0:
		raise Exception("chunklength_in_sec must be different than 0")
	elif chunklength_in_sec == None:
		return [(starttime,endtime)]

	if overlap_in_sec == None:
		overlap_in_sec = 0

	deltat = starttime
	dtt = dt.timedelta(seconds=chunklength_in_sec)
	overlap_dt = dt.timedelta(seconds=overlap_in_sec)

	times = []
	while deltat < endtime:
		# chunklength can't be greater than (endtime-startime)
		if deltat + dtt > endtime:
			break
		else:
			times.append((deltat,deltat+dtt))
			deltat += dtt - overlap_dt

	if deltat < endtime:	
		times.append((deltat,endtime))
	# print(times)
	return times

if __name__ == "__main__":
    path = "/home/emmanuel/ecastillo/dev/delaware/data/metadata/delaware_stations_160824.csv"
    df = pd.read_csv(path)
    filter_info(df,remove_networks=[],remove_stations=[],
                location_pref=["20"],
                # location_pref=["00","10","20"],
                        instrument_pref=["HH","EH"],
                        # domain=[-94,-95,31,32]
                        )
    
    
    # print(df)