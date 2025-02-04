import numpy as np
import pandas as pd
import os
import numpy as np
import concurrent.futures as cf
from tqdm import tqdm
from obspy import UTCDateTime
from utdquake.core.database.database import save_to_sqlite

class StatValues:
    def __init__(self,
                 availability=np.nan,
                 gaps_duration=np.nan,
                 overlaps_duration=np.nan,
                 gaps_counts=np.nan,
                 overlaps_counts=np.nan) -> None:
        """
        Initialize acquisition quality control data.

        Parameters:
        -----------
        availability : float, optional
            The availability percentage of the data. Default is NaN.
        gaps_duration : float, optional
            The total duration of gaps in the data. Default is NaN.
        overlaps_duration : float, optional
            The total duration of overlaps in the data. Default is NaN.
        gaps_counts : int, optional
            The count of gaps in the data. Default is NaN.
        overlaps_counts : int, optional
            The count of overlaps in the data. Default is NaN.
        """
        self.availability = availability
        self.gaps_duration = gaps_duration
        self.overlaps_duration = overlaps_duration
        self.gaps_counts = gaps_counts
        self.overlaps_counts = overlaps_counts
    
    def __str__(self, extended=False) -> str:
        """
        Return a string representation of the statistical values.

        Parameters:
        -----------
        extended : bool, optional
            If True, provides detailed information. Default is False.

        Returns:
        --------
        str
            A formatted string describing the statistical values.
        """
        if extended:
            msg = (f"Stats | ({self.availability} %)"
                   + f"\n\tAvailability: {self.availability}"
                   + f"\n\tGaps duration: {self.gaps_duration}"
                   + f"\n\tGaps count: {self.gaps_counts}"
                   + f"\n\tOverlaps duration: {self.overlaps_duration}"
                   + f"\n\tOverlaps count: {self.overlaps_counts}")
        else:
            msg = f"Stats | (Availability: {self.availability} %)"
        
        return msg
    
    def to_dict(self):
        return {"availability":self.availability,
                "gaps_duration": self.gaps_duration,
                "gaps_counts":self.gaps_duration,
                "overlaps:_duration":self.overlaps_duration,
                "overlap_counts":self.overlaps_counts}

    @property
    def empty(self):
        if self.availability == np.nan:
            return False
        else:
            return True

def get_stats_by_instrument(instrument_strid, stream,
                             channels=["HHZ", "HHE", "HHN"],
                             starttime=None, endtime=None,
                             debug=True, workers=1):
    """
    Calculate statistics for seismic data from specified channels and time range.

    Args:
        instrument_strid (str): The identifier string for the instrument, formatted as "NET.STA.LOC.CHA".
        stream (Stream): The Obspy Stream object containing seismic data.
        channels (list of str): List of channel codes to process (default is ["HHZ", "HHE", "HHN"]).
        starttime (str or None): The start time for the data (format: 'YYYY-MM-DDTHH:MM:SS'). If None, use the earliest available.
        endtime (str or None): The end time for the data (format: 'YYYY-MM-DDTHH:MM:SS'). If None, use the latest available.
        debug (bool): If True, print debug information (default is True).
        workers (int): The number of parallel workers to use for processing (default is 3).

    Returns:
        pd.DataFrame: A DataFrame containing statistics for each channel, including:
            - availability: Percentage of time data is available.
            - gaps_duration: Total duration of gaps in the data.
            - overlaps_duration: Total duration of overlaps in the data.
            - gaps_counts: Number of gaps detected.
            - overlaps_counts: Number of overlaps detected.
    """
    
    # Check if stream is provided
    if not stream:
        raise ValueError("No stream provided")

    stats_dict = {}

    def _get_stats_by_channel(channel):
        """
        Calculate statistics for a specific channel within the given time range.

        Args:
            channel (str): The channel code to process.
        """
        strid = ".".join((instrument_strid, channel))
        net, sta, loc, cha = strid.split(".")

        if debug:
            print(f"Processing channel: {strid}")

        # Select the data for the specified channel
        st = stream.copy().select(network=net, station=sta,
                                  location=loc, channel=cha)

        if not st:
            stats_dict[strid] = StatValues().to_dict()
            return

        st.sort(keys=['starttime', 'endtime'])
        st.traces = [tr for tr in st
                     if not (tr.stats.endtime < starttime or
                             tr.stats.starttime > endtime)]

        # Set start and end times
        cha_starttime = UTCDateTime(starttime) if starttime else st[0].stats.starttime
        cha_endtime = UTCDateTime(endtime) if endtime else st[-1].stats.endtime

        total_duration = cha_endtime - cha_starttime

        # Initialize gap and overlap statistics
        gaps = []
        overlaps = []
        gap_count = 0
        overlap_count = 0

        # Compute gaps and overlaps
        for gap in st.get_gaps():
            if gap[6] > 0:
                gap_count += 1
                gaps.append(gap[6])
            else:
                overlap_count += 1
                overlaps.append(gap[6])

        # Sum up gaps and overlaps
        gap_sum = np.sum(gaps)
        overlap_sum = np.sum(overlaps)

        if not st:
            stats_dict[strid] = StatValues().to_dict()
            return
        
        # Check for gaps at the start or end
        earliest = min([tr.stats.starttime for tr in st])
        latest = max([tr.stats.endtime for tr in st])
        if earliest > cha_starttime:
            gap_sum += earliest - cha_starttime
            gap_count += 1
        if latest < cha_endtime:
            gap_sum += cha_endtime - latest
            gap_count += 1

        # Calculate percentage of available data
        percentage = 100 * (1 - (gap_sum / total_duration))
        percentage = round(percentage, 2)

        # Store the statistics
        stat_args = {
            "availability": percentage,
            "gaps_duration": gap_sum,
            "overlaps_duration": overlap_sum,
            "gaps_counts": gap_count,
            "overlaps_counts": overlap_count,
        }
        stats_dict[strid] = stat_args

    # Process channels either sequentially or in parallel
    if workers == 1:
        if debug:
            channels = tqdm(channels)
        for channel in channels:
            _get_stats_by_channel(channel)
    else:
        with cf.ThreadPoolExecutor(max_workers=workers) as executor:
            if debug:
                list(tqdm(executor.map(_get_stats_by_channel, channels),
                          total=len(channels)))
            else:
                executor.map(_get_stats_by_channel, channels)

    # Convert the results to a DataFrame and sort
    stats = pd.DataFrame.from_dict(stats_dict)
    stats.index.name = "stats"

    columns = stats.columns.to_list()
    columns.sort()
    stats = stats[columns]
    stats.sort_index(inplace=True)

    return stats

def get_stats(st, starttime=None, endtime=None, debug=True, workers=1,
              ):
    """
    Calculate statistics for seismic data grouped by instrument.

    Args:
        st (Stream): The Obspy Stream object containing seismic data.
        starttime (str or None): The start time for the data (format: 'YYYY-MM-DDTHH:MM:SS'). If None, use the earliest available.
        endtime (str or None): The end time for the data (format: 'YYYY-MM-DDTHH:MM:SS'). If None, use the latest available.
        debug (bool): If True, print debug information (default is True).
        workers (int): The number of parallel workers to use for processing (default is 1).

    Returns:
        pd.DataFrame: A DataFrame containing statistics for each instrument, including:
            - Availability percentage
            - Gaps duration
            - Overlaps duration
            - Gaps count
            - Overlaps count
    """
    
    # Check if stream is provided
    if not st:
        raise ValueError("No stream provided")

    # Group the stream by network, station, and location
    instrument_dict = st._groupby("{network}.{station}.{location}").items()
    stats = []

    # Convert instrument_dict to a list of tuples
    instrument_items = list(instrument_dict)

    def _get_stats_by_instrument(instrument_item):
        """
        Calculate statistics for a specific instrument.

        Args:
            instrument_item (tuple): A tuple containing the instrument ID and the corresponding Stream.
        """
        instrument_strid, instrument_st = instrument_item
        
        # Get common channels information
        instrument_info = instrument_st._get_common_channels_info()
        channels = [list(var["channels"].keys()) for var in instrument_info.values()]
        channels = [item for sublist in channels for item in sublist]
        channels = list(set(channels))
        
        # Get statistics for the instrument
        stats_by_instrument = get_stats_by_instrument(instrument_strid, instrument_st,
                                                      channels=channels,
                                                      starttime=starttime, endtime=endtime,
                                                      debug=False, workers=1)
        stats.append(stats_by_instrument)

    # Process instruments either sequentially or in parallel
    if workers == 1:
        if debug:
            instrument_items = tqdm(instrument_items)
        for instrument_item in instrument_items:
            _get_stats_by_instrument(instrument_item)
    else:
        with cf.ThreadPoolExecutor(max_workers=workers) as executor:
            if debug:
                list(tqdm(executor.map(_get_stats_by_instrument, instrument_items),
                          total=len(instrument_items)))
            else:
                executor.map(_get_stats_by_instrument, instrument_items)

    # Concatenate statistics DataFrames and sort
    stats = pd.concat(stats, axis=1)

    # Ensure columns are in order and index is sorted
    columns = stats.columns.to_list()
    columns.sort()
    stats = stats[columns]
    stats.sort_index(inplace=True)

    return stats

def get_rolling_stats(st, step=3600, starttime=None, endtime=None, 
                      sqlite_output=None):
    """
    Calculate rolling statistics for seismic data over specified time intervals.

    Parameters:
    -----------
    st : Stream
        The ObsPy Stream object containing seismic data.
    step : int, optional
        The step size for the rolling window in seconds. Default is 3600 seconds (1 hour).
    starttime : str or None, optional
        The start time for the data in 'YYYY-MM-DDTHH:MM:SS' format. If None, use the start of the data stream.
    endtime : str or None, optional
        The end time for the data in 'YYYY-MM-DDTHH:MM:SS' format. If None, use the end of the data stream.
    sqlite_output : str or None, optional
        Path to the SQLite database folder for saving the results. If None, results are not saved to SQLite.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing rolling statistics for each interval, with columns including:
            - Availability percentage
            - Gaps duration
            - Overlaps duration
            - Gaps count
            - Overlaps count
    """
    
    # Retrieve common channel information from the stream and convert it to a DataFrame
    list_of_dicts = [
        {
            'network': v[0],
            'station': v[1],
            'location': v[2],
            'instrument': v[3][0:2]
        }
        for v in st._get_common_channels_info().keys()
    ]
    common_channels = pd.DataFrame(list_of_dicts)
    
    # Sort the stream by starttime and endtime
    st.sort(keys=['starttime', 'endtime'])

    # Determine the start and end times for the rolling window
    if starttime is None:
        starttime = st[0].stats.starttime.datetime
    if endtime is None:
        endtime = st[-1].stats.endtime.datetime

    # Generate a range of dates with the specified step size
    dates = pd.date_range(start=starttime, end=endtime, freq=f"{step}s")

    all_stats = []
    
    str_id = ".".join((st[0].stats.network,
                st[0].stats.station,
                st[0].stats.location,
                st[0].stats.channel[0:2]
            ))

    # Iterate over each time window and compute statistics
    for i in tqdm(range(len(dates) - 1), desc=f"Processing intervals | {str_id}"):
        interval_starttime = dates[i]
        interval_endtime = dates[i + 1]
        
        # Compute statistics for the current time window
        stats = get_stats(st=st, starttime=interval_starttime, endtime=interval_endtime, debug=False)
        
        # Add start and end times to the statistics DataFrame
        stats["starttime"] = interval_starttime
        stats["endtime"] = interval_endtime
        stats.set_index(["starttime", "endtime"], inplace=True, append=True)
        
        # Append the statistics to the list
        all_stats.append(stats)
    
    # Concatenate all statistics DataFrames and sort
    all_stats = pd.concat(all_stats, axis=0)
    columns = all_stats.columns.to_list()
    columns.sort()
    all_stats = all_stats[columns]
    all_stats.sort_index(inplace=True)

    # Reset index for saving to SQLite
    all_stats = all_stats.reset_index()

    # Save the DataFrame to SQLite if a path is provided
    if sqlite_output is not None:
        if not os.path.isdir(sqlite_output):
            os.makedirs(sqlite_output)
        
        # Extract unique statistics keys
        stats_keys = all_stats.drop_duplicates(subset="stats")["stats"].to_list()
        
        # Iterate over common channels and save statistics to SQLite
        for _, value in common_channels.iterrows():
            str_id = ".".join((value.network, value.station,
                               value.location, value.instrument))

            # Filter columns that match the pattern
            filtered_columns = [col for col in all_stats.columns if col.startswith(str_id)]
            components = [x[-1] for x in filtered_columns]
            comp_str = "".join(components)
            db_path = os.path.join(sqlite_output, str_id + f"_{comp_str}.db")

            # Create a new DataFrame with only the filtered columns
            stats_per_channel = all_stats[['stats', 'starttime', 'endtime'] + filtered_columns]
            
            # Save each subset of statistics to the SQLite database
            for key in stats_keys:
                stats = stats_per_channel[stats_per_channel["stats"] == key]
                stats = stats.drop("stats", axis=1)
                save_to_sqlite(stats, db_path, key)

    return all_stats





if __name__ == "__main__":
    from obspy import read, UTCDateTime
    from core.client import LocalClient
    import os 
    import pandas as pd
    
    # archive = r"/home/emmanuel/ecp_archive/APIAY/seedfiles"
    # archive_fmt = os.path.join("{year}-{month:02d}", 
    #                 "{year}-{month:02d}-{day:02d}", 
    #                 "{network}.{station}.{location}.{channel}.{year}.{julday:03d}")
    # client = LocalClient(archive,archive_fmt)

    # st = client.get_waveforms(network="EY",
    #                     station="AP0[12]B",
    #                     location="*",
    #                     channel="HH*",
    #                     starttime=UTCDateTime("2024-08-06T00:00:00"),
    #                     endtime=UTCDateTime("2024-08-06T12:00:00"))
    # print(st)
    
    
    # # st = read("/home/emmanuel/ecp_archive/APIAY/seedfiles/2024-08/2024-08-06/EY.AP01B.00.HHE.2024.219")
    # starttime = UTCDateTime("2024-08-06T00:00:00")
    # endtime = UTCDateTime("2024-08-06T12:00:00")
    
    
    # plot_rolling_stats(stats=)
    # stats = get_rolling_stats(st, step=3600, 
    #                           starttime=starttime.datetime, 
    #                           endtime=endtime.datetime,
    #                           sqlite_output=None)
    # print(stats.loc["availability"])