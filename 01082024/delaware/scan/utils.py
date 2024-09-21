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
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import os

def get_db_paths(providers, db_folder_path, network, station, location, instrument):
    """
    Generate a list of database paths that match the specified criteria.

    Args:
        providers (list): List of provider objects, each with an 'info' attribute containing station information.
        db_folder_path (str): Path to the folder containing database files.
        network (str): Network identifier.
        station (str): Station identifier.
        location (str): Location identifier.
        instrument (str): Instrument identifier.

    Returns:
        list: List of paths to the database files that match the given criteria.
    """
    # Construct the base name for the database files
    db_name = ".".join((network, station, location, instrument))
    
    # Create the search pattern for the database files
    key = os.path.join(db_folder_path, db_name + "**")
    
    # Find all files that match the search pattern
    db_paths = glob.glob(key)
    
    # Extract the unique station identifiers from the database paths
    stainpaths = [x.split(".")[1] for x in db_paths]
    
    # Gather all unique stations from the provider information
    stations2scan = []
    for provider in providers:
        prov_info = provider.info["station"].drop_duplicates()
        for prov_sta in prov_info.tolist():
            stations2scan.append(prov_sta)
    
    # Find the intersection of station identifiers between providers and database paths
    intersection = list(set(stations2scan) & set(stainpaths))
    
    # Filter the database paths to include only those with stations in the intersection
    db_paths = [x for x in db_paths if x.split(".")[1] in intersection]
    
    return db_paths
    
class StatsColorBar:
    """
    A class to create and manage a color bar for visualizing statistical data.

    Attributes:
        stat (str): The type of statistic being visualized (e.g., "availability").
        label_dict (dict): Dictionary mapping labels to value ranges.
        cmap_name (str): Name of the colormap to use for the color bar.
        bad_colorname (str): Color to use for "bad" or missing values.
        zero_colorname (str): Color to use for zero values.
    """
    def __init__(self, stat, label_dict, cmap_name="viridis_r", bad_colorname="gray", zero_colorname="white") -> None:
        """
        Initializes the StatsColorBar with the given parameters.

        Args:
            stat (str): The type of statistic being visualized.
            label_dict (dict): Dictionary mapping labels to value ranges.
            cmap_name (str): Name of the colormap.
            bad_colorname (str): Color for bad or missing values.
            zero_colorname (str): Color for zero values.
        """
        self.stat = stat
        self.label_dict = label_dict
        self.cmap_name = cmap_name
        self.bad_colorname = bad_colorname
        self.zero_colorname = zero_colorname

    def get_label_dict(self):
        """
        Retrieves the label dictionary. If not provided, generates a default one based on the statistic type.

        Returns:
            dict: Dictionary mapping labels to their corresponding value ranges.

        Raises:
            Exception: If `label_dict` is not provided and `stat` is not "availability".
        """
        if self.label_dict is None:
            if self.stat == "availability":
                # label_dict = {
                #     "No gaps": [0, 1e-5],
                #     r"$\leq 1$ s": [1e-5, 1],
                #     r"$\leq 1$ min": [1, 60],
                #     r"$\leq 1$ hour": [60, 60**2],
                #     r"$< 1$ day": [60**2, (60**3) * 24],
                #     r"$\geq 1$ day": [(60**3) * 24, ((60**3) * 24) + 1]
                # }
                label_dict={"[0,20)":[0,20],
                            r"[20,40)":[20,40],
                            r"[40,60)":[40,60],
                            r"[60,80)":[60,80],
                            r"[80,100]":[80,100],
                            }
            else:
                raise Exception("label_dict must be specified for the given statistic type.")
        else:
            label_dict = self.label_dict
        return label_dict

    def get_colorbar_info(self, cmap_name="viridis_r", bad_colorname="gray", zero_colorname="white"):
        """
        Generates information required to create a color bar including colormap, formatter, ticks, and normalization.

        Args:
            cmap_name (str): Name of the colormap.
            bad_colorname (str): Color for bad or missing values.
            zero_colorname (str): Color for zero values.

        Returns:
            dict: Dictionary containing colormap, format function, ticks, and normalization.
        """
        # Retrieve the label dictionary
        label_dict = self.get_label_dict()
        
        # Extract labels and boundaries from the dictionary
        labels = list(label_dict.keys())
        flattened_list = [item for sublist in label_dict.values() for item in sublist]
        boundaries = sorted(list(set(flattened_list)))
        
        # Create the colormap with the number of colors needed
        ncolors = len(boundaries) - 1
        cmap = plt.get_cmap(cmap_name, ncolors)

        # Set the color for bad values
        if bad_colorname is not None:
            cmap.set_bad(color=bad_colorname)
        
        # Retrieve colors from the colormap
        colors = list(cmap(np.arange(ncolors)))

        # Set the color for zero values
        if zero_colorname is not None:
            colors[0] = zero_colorname
        
        # Create a colormap from the list of colors
        from_list = mpl.colors.LinearSegmentedColormap.from_list
        cm = from_list(None, colors, ncolors)

        # Create a normalization object for the colorbar
        norm = mpl.colors.BoundaryNorm(boundaries, ncolors=ncolors, clip=True)

        # Create a formatter for the colorbar ticks
        fmt = mpl.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
        
        # Calculate tick positions
        boundaries = np.array(boundaries)
        diff = boundaries[1:] - boundaries[:-1]
        ticks = boundaries[:-1] + diff / 2

        # Compile all the colorbar information into a dictionary
        colorbar_info = {
            "cmap": cm,
            "format": fmt,
            "ticks": ticks,
            "norm": norm
        }
        return colorbar_info





def sort_yaxis_info(stat, perc=True):
    """
    Sorts and organizes the y-axis information for a plot based on statistical data.

    Args:
        stat (pd.DataFrame): DataFrame containing statistical data, with columns formatted as 'network.station.location.channel'.
        perc (bool): If True, calculates the percentage availability of each column. Defaults to True.

    Returns:
        dict: Dictionary containing sorted y-axis labels, availability information, tick positions, and column order.
    """
    # Initialize dictionaries to group by station and location
    group = {"station": [], "location": []}
    strid_names = {}
    y_availability = []
    y_mask = []
    
    # Custom sorting function to extract and sort based on network, station, location, and channel
    def sort_key(item):
        network, station, location, channel = item.split('.')
        return (network, station, location, channel)

    # Retrieve and sort columns based on the custom sort key
    columns = stat.columns.to_list()
    columns = sorted(columns, key=sort_key)
    
    if perc:
        # Calculate mean availability for each column, handle NaN values by filling with 0
        availability = stat[columns].mean()
        availability = availability.to_dict()
    else:
        availability = {}

    # Iterate over sorted columns to build the y-axis information
    for i, strid in enumerate(columns):
        net, sta, loc, cha = strid.split('.')
        new_strid = strid
        
        # Check if the station has already been processed
        if net + "." + sta in group["station"]:
            new_strid = ".".join((loc, cha))
        else:
            group["station"].append(net + "." + sta)
            y_mask.append(len(columns) - i)
        
        # Check if the location has already been processed
        if net + "." + sta + "." + loc in group["location"]:
            new_strid = cha
        else:
            group["location"].append(net + "." + sta + "." + loc)
        
        strid_names[strid] = new_strid

        # Append availability information if requested
        if availability:
            y_availability.append(f"{round(availability[strid], 1)}%")

    # Prepare y-axis labels, mask, and order
    y_names = [strid_names[strid] for strid in columns]
    y_mask = np.array(y_mask + [0])  # Append 0 to ensure the mask covers all ticks

    # Compile the y-axis information into a dictionary
    yaxis_info = {
        "labels": y_names,
        "availability": y_availability,
        "ticks": y_mask,
        "order": columns
    }
    
    return yaxis_info

def sort_xaxis_info(stat, major_step, major_format="%Y-%m-%d %H:%M:%S"):
    """
    Sorts and organizes x-axis information for a plot based on statistical data with datetime indices.

    Args:
        stat (pd.DataFrame): DataFrame with a MultiIndex where one level is 'starttime' and another level is 'endtime'.
        major_step (int): Number of intervals between major ticks, specified in seconds.
        major_format (str, optional): Format string for major tick labels. Defaults to "%Y-%m-%d %H:%M:%S".

    Returns:
        dict: Dictionary containing lists of minor and major dates for x-axis ticks.
    """
    # Retrieve the start and end times from the DataFrame index
    start_times = stat.index.get_level_values('starttime').to_list()
    end_times = stat.index.get_level_values('endtime').to_list()

    # Combine start times and the last end time to create a continuous list of minor ticks
    minor_dates = start_times + [end_times[-1]]

    # Generate major ticks by rounding minor ticks to the nearest second and selecting every `major_step` interval
    major_dates = [
        minor_dates[i].round("s")
        for i in range(0, len(minor_dates), major_step)
    ]

    # Convert minor and major dates to Python datetime objects and format major dates
    minor_dates = [ts.to_pydatetime() for ts in minor_dates]
    major_dates = [ts.to_pydatetime().strftime(major_format) for ts in major_dates]

    # Compile the x-axis information into a dictionary
    xaxis_info = {
        "minor": minor_dates,
        "major": major_dates
    }

    return xaxis_info

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
        for v in st.copy().merge()._get_common_channels_info().keys()
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