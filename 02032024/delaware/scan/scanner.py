# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-08-17 11:16:32
#  * @modify date 2024-08-17 11:16:32
#  * @desc [description]
#  */
import utdquake.scan.utils as ut
from utdquake.tools.stats import get_rolling_stats
from utdquake.core.database.database import load_from_sqlite

import logging
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import concurrent.futures as cf
from obspy import UTCDateTime

logger = logging.getLogger("utdquake.scan.scanner")

class Provider:
    """
    A class to manage and query seismic data from a provider with specific restrictions.

    Attributes:
    -----------
    client : Client
        The client object used to interact with the data provider.
    wav_restrictions : Restrictions
        The restrictions for querying data such as network, station, and instrument preferences.
    """

    def __init__(self, client, wav_restrictions) -> None:
        """
        Initialize the Provider with a client and wave restrictions.

        Parameters:
        -----------
        client : Client
            The client object used for data queries.
        wav_restrictions : Restrictions
            The restrictions for querying data.
        """
        self.client = client
        self.wav_restrictions = wav_restrictions

    def __str__(self, extended=False) -> str:
        """
        Return a string representation of the Provider.

        Parameters:
        -----------
        extended : bool, optional
            If True, include extended information in the string representation.

        Returns:
        --------
        str
            A formatted string describing the provider and its restrictions.
        """
        msg = (f"Provider: {self.client.base_url}  "
               f"\n\tRestrictions: {self.wav_restrictions.__str__(extended)}")
        return msg

    @property
    def inventory(self):
        """
        Retrieve the inventory of stations from the client based on the wave restrictions.

        Returns:
        --------
        Inventory
            The inventory of stations, channels, and locations.
        """
        inventory = self.client.get_stations(
            network=self.wav_restrictions.network,
            station=self.wav_restrictions.station,
            location="*",
            channel="*",
            level='channel'
        )
        return inventory

    @property
    def info(self):
        """
        Get the filtered information based on the inventory and wave restrictions.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the filtered inventory information.
        """
        info = ut.get_inventory_info(self.inventory)
        wr = self.wav_restrictions

        # Filter the information based on restrictions
        info = ut.filter_info(
            info,
            remove_networks=wr.remove_networks,
            remove_stations=wr.remove_stations,
            location_pref=wr.location_preferences,
            instrument_pref=wr.instrument_preferences,
            handle_preference="sort",
            domain=wr.filter_domain
        )
        return info

    def get_info_to_query(self, level="station"):
        """
        Prepare query information based on the specified level.

        Parameters:
        -----------
        level : str, optional
            The level of detail for the query. Options are "station", "instrument", or "channel".

        Returns:
        --------
        list of tuples
            A list of tuples where each tuple contains query parameters for the specified level.

        Raises:
        -------
        Exception
            If the level is not one of "station", "instrument", or "channel".
        """
        info = self.info
        station_level = ["network", "station"]

        if level == "station":
            # Prepare query for station level
            info = info.drop_duplicates(subset=station_level)
            i2q = list(zip(
                info["network"].tolist(),
                info["station"].tolist(),
                ["*"] * len(info),
                ["*"] * len(info)
            ))
        elif level == "instrument":
            # Prepare query for instrument level
            info = info.drop_duplicates(subset=station_level + ["instrument"])
            i2q = list(zip(
                info["network"].tolist(),
                info["station"].tolist(),
                info["location_code"].tolist(),
                info["channel"].tolist()
            ))
        elif level == "channel":
            # Prepare query for channel level
            info = info.drop_duplicates(subset=station_level + ["channel"])
            i2q = list(zip(
                info["network"].tolist(),
                info["station"].tolist(),
                info["location_code"].tolist(),
                info["channel"].tolist()
            ))
        else:
            raise Exception("Available levels are 'station', 'instrument', or 'channel'")

        return i2q
            
class WaveformRestrictions:
    """
    A class to define restrictions for querying waveform data.

    Attributes:
    -----------
    network : str
        One or more network codes, comma-separated. Wildcards are allowed.
    station : str
        One or more SEED station codes, comma-separated. Wildcards are allowed.
    location : str
        One or more SEED location identifiers, comma-separated. Wildcards are allowed.
    channel : str
        One or more SEED channel codes, comma-separated.
    starttime : obspy.UTCDateTime
        Limit results to time series samples on or after this start time.
    endtime : obspy.UTCDateTime
        Limit results to time series samples on or before this end time.
    location_preferences : list
        List of location preferences in order. Only the first element's location will be selected.
    instrument_preferences : list
        List of instrument preferences.
    remove_networks : list
        List of networks to be excluded.
    remove_stations : list
        List of stations to be excluded.
    filter_domain : list
        Geographic domain for filtering results in the format [lonw, lone, lats, latn].
    minimumlength: int
            Limit results to continuous data segments of a minimum length specified in seconds.
    """

    def __init__(self, network, station, location, channel, starttime, endtime,
                 location_preferences=["", "00", "20", "10", "40"],
                 instrument_preferences=["HH", "BH", "EH", "HN", "HL"],
                 remove_networks=[], remove_stations=[], 
                 filter_domain=[-180, 180, -90, 90],
                 minimumlength=None):
        """
        Initialize the WaveformRestrictions with specified parameters.

        Parameters:
        -----------
        network : str
            Select one or more network codes. Wildcards are allowed.
        station : str
            Select one or more SEED station codes. Wildcards are allowed.
        location : str
            Select one or more SEED location identifiers. Wildcards are allowed.
        channel : str
            Select one or more SEED channel codes.
        starttime : obspy.UTCDateTime
            Limit results to time series samples on or after this start time.
        endtime : obspy.UTCDateTime
            Limit results to time series samples on or before this end time.
        location_preferences : list, optional
            List of locations in order of preference. Defaults to ["", "00", "20", "10", "40"].
        instrument_preferences : list, optional
            List of instrument preferences. Defaults to ["HH", "BH", "EH", "HN", "HL"].
        remove_networks : list, optional
            List of networks to exclude. Defaults to an empty list.
        remove_stations : list, optional
            List of stations to exclude. Defaults to an empty list.
        filter_domain : list, optional
            Geographic domain for filtering in the format [lonw, lone, lats, latn]. Defaults to [-180, 180, -90, 90].
        minimumlength: int
            Limit results to continuous data segments of a minimum length specified in seconds.
        """
        self.network = network
        self.station = station
        self.location = location
        self.channel = channel
        self.starttime = starttime
        self.endtime = endtime
        self.location_preferences = location_preferences
        self.instrument_preferences = instrument_preferences
        self.remove_networks = remove_networks
        self.remove_stations = remove_stations
        self.filter_domain = filter_domain
        self.minimumlength = minimumlength

    def __str__(self, extended=False) -> str:
        """
        Return a string representation of the WaveformRestrictions.

        Parameters:
        -----------
        extended : bool, optional
            If True, include detailed information. Defaults to False.

        Returns:
        --------
        str
            A formatted string describing the waveform restrictions.
        """
        timefmt = "%Y%m%dT%H:%M:%S"
        if extended:
            msg = (f"Waveform Restrictions"
                   f"\n\tnetwork: {self.network}"
                   f"\n\tstation: {self.station}"
                   f"\n\tlocation: {self.location}"
                   f"\n\tchannel: {self.channel}"
                   f"\n\tstarttime: {self.starttime.strftime(timefmt)}"
                   f"\n\tendtime: {self.endtime.strftime(timefmt)}"
                   f"\n\tlocation_preferences: {self.location_preferences}"
                   f"\n\tinstrument_preferences: {self.instrument_preferences}"
                   f"\n\tremove_networks: {self.remove_networks}"
                   f"\n\tremove_stations: {self.remove_stations}"
                   f"\n\tfilter_domain: {self.filter_domain}",
                   f"\n\tminimumlength: {self.minimumlength}")
        else:
            msg = (f"Waveform Restrictions"
                   f"\n\t{self.network}.{self.station}.{self.location}.{self.channel}|"
                   f"{self.starttime.strftime(timefmt)}-{self.endtime.strftime(timefmt)}")
        return msg

class Scanner(object):
    """
    A class to scan waveform data based on specified providers and parameters.

    Attributes:
    -----------
    db_folder_path : str
        Path to the SQLite database folder for saving the results.
    providers : list
        List of FDSN client instances or service URLs.
    configure_logging : bool
        Flag to configure logging on initialization. Defaults to True.
    """

    def __init__(self, db_folder_path, providers=[], configure_logging=True):
        """
        Initialize the Scanner with a database path, list of providers, and optional logging configuration.

        Parameters:
        -----------
        db_folder_path : str
            Path to the SQLite database folder for saving the results.
        providers : list, optional
            List of FDSN client instances or service URLs. Defaults to an empty list.
        configure_logging : bool, optional
            Flag to configure logging. Defaults to True.
        """
            
        self.logging_path = None
        
        if configure_logging:
            self._setup_logging(db_folder_path)

        self.db_folder_path = db_folder_path
        self.providers = providers

    def _setup_logging(self, db_folder_path):
        """
        Set up logging configuration for the Scanner.

        Parameters:
        -----------
        db_folder_path : str
            Path to the SQLite database folder used to determine the logging folder.
        """
        logging_folder_path = os.path.join(os.path.dirname(db_folder_path),
                                           os.path.basename(db_folder_path) + "_log")
        if not os.path.isdir(logging_folder_path):
            os.makedirs(logging_folder_path)

        timenow = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.logging_path = os.path.join(logging_folder_path, f"ScannerLog_{timenow}.log")

        # Create a logger instance for this class
        logger.setLevel(logging.DEBUG)  # Set the log level to DEBUG for the logger

        # Console log handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # Console handler logs INFO level and above

        # File log handler
        fh = logging.FileHandler(self.logging_path)
        fh.setLevel(logging.DEBUG)  # File handler logs DEBUG level and above

        # Formatter for log messages
        formatter = logging.Formatter("[%(asctime)s] - %(name)s - %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(ch)
        logger.addHandler(fh)

        # Prevent log messages from being passed to higher loggers
        logger.propagate = 0

    def scan(self, step, wav_length=86400, max_traces=1000, level="station", n_processor=1):
        """
        Scan the waveform data for each provider and save results to the database.

        Parameters:
        -----------
        step : int
            The step size for rolling statistics calculation.
        wav_length : int, optional
            Length of each waveform chunk in seconds. Defaults to 86400 seconds (1 day).
        max_traces: int,
            Maximum number of traces allowed per request. It prevents to spend a lot of time in corrupted data.
        level : str, optional
            Level of information to query. Options are "station", "instrument", or "channel". Defaults to "station".
        n_processor : int, optional
            Number of parallel processors to use. Defaults to 1 for no parallelism.
        """
        for provider in self.providers:
            logger.info(f"{provider}")

            starttime = provider.wav_restrictions.starttime
            endtime = provider.wav_restrictions.endtime

            # Generate chunk times for querying
            times = ut.get_chunktimes(starttime=starttime,
                                    endtime=endtime,
                                    chunklength_in_sec=wav_length,
                                    overlap_in_sec=0)
            logger.info(f"Number of queries per provider: {len(times)}")

            # Get query information based on the desired level
            i2q = provider.get_info_to_query(level=level)

            logger.info(f"Number of queries per {level}: {len(i2q)}")
            logger.info(f"Total number of queries: {len(times) * len(i2q)}")

            for chunk_starttime, chunk_endtime in times:
                logger.info(f"{'#'*12} Starttime: {chunk_starttime} - Endtime: {chunk_endtime} {'#'*12}")

                client = provider.client
                wr = provider.wav_restrictions

                def scan_query(info):
                    """
                    Query the waveform data and process it.

                    Parameters:
                    -----------
                    info : tuple
                        A tuple containing network, station, location, and channel codes.
                    """
                    net, sta, loc, cha = info
                    logger.info(f"Loading the stream: {info}|{chunk_starttime}-{chunk_endtime}")

                    try:
                        # Fetch waveform data from the client
                        st = client.get_waveforms(network=net,
                                                station=sta,
                                                location=loc,
                                                channel=cha,
                                                starttime=chunk_starttime,
                                                endtime=chunk_endtime,
                                                minimumlength=wr.minimumlength)
                    except Exception as e:
                        logger.error(f"{info}|{chunk_starttime}-{chunk_endtime}"+f"\n{e}")
                        st = False

                    if not st:
                        logger.error(f"{info}|{chunk_starttime}-{chunk_endtime}"+f"\tNo strem to process.")
                        return
                    elif len(st) > max_traces:
                        logger.error(f"{info}|{chunk_starttime}-{chunk_endtime}"+\
                            f"\tStream no considered because exceeds number of traces allowed: {len(st)}/{max_traces}")
                        return
                    

                    try:
                        logger.info(f"Checking the stream: {info}|{chunk_starttime}-{chunk_endtime}")
                        # Process the stream to standardize channels
                        st = ut.process_stream_common_channels(st,
                                                            location_preferences=wr.location_preferences,
                                                            instrument_preferences=wr.instrument_preferences)
                        logger.info(f"Scanning the stream: {info}|{chunk_starttime}-{chunk_endtime}")
                        # Compute and save rolling statistics
                        get_rolling_stats(st, step=step,
                                        starttime=chunk_starttime.datetime,
                                        endtime=chunk_endtime.datetime,
                                        sqlite_output=self.db_folder_path)
                    except Exception as e:
                        logger.error(f"{info}|{chunk_starttime}-{chunk_endtime}"+f"\n{e}")
                        return

                # Perform the query for each set of parameters
                if n_processor == 1:
                    for info in i2q:
                        scan_query(info)
                else:
                    with cf.ThreadPoolExecutor(n_processor) as executor:
                        executor.map(scan_query, i2q)
                 
    def get_stats(self, network, station, location, instrument,
              starttime, endtime, stats=["availability", "gaps_counts"]):
        """
        Retrieve statistical data from database files based on the provided criteria.

        Args:
            network (str): Network identifier.
            station (str): Station identifier.
            location (str): Location identifier.
            instrument (str): Instrument identifier.
            starttime (datetime): Start time for the data retrieval.
            endtime (datetime): End time for the data retrieval.
            stats (list): List of statistical metrics to retrieve from the database.

        Returns:
            pd.DataFrame: DataFrame containing concatenated statistical data, or None if no data is found.
        """
        if isinstance(starttime, UTCDateTime):
            starttime = starttime.datetime
        if isinstance(endtime, UTCDateTime):
            endtime = endtime.datetime
        
        # Get the database paths matching the criteria
        db_paths = ut.get_db_paths(providers=self.providers,
                                db_folder_path=self.db_folder_path,
                                network=network, station=station,
                                location=location, instrument=instrument)
        
        if not db_paths:
            logger.info(f"No paths found using the provided key in the glob search.")
        
        logger.info(f"Loading: {len(db_paths)} paths")
        
        all_dfs = {}
        
        # Process each database path
        for i, db_path in enumerate(db_paths, 1):
            logger.info(f"Loading: {i}/{len(db_paths)} {db_path}")
            
            # Extract network and station path components
            netstainpath = os.path.basename("_".join(db_path.split(".")[0:2]))
            
            dfs_stats = []
            
            # TODO: I can improve this, I am loading table by table
            # and I can actually do all in one
             
            # Load statistics for each specified metric
            for stat in stats:
                
                try:
                    custom_params = {"starttime":{"condition":">=","value":starttime},
                                     "endtime":{"condition":"<","value":endtime}}
                    df = load_from_sqlite(db_path=db_path,
                                                    tables=[stat],
                                                    custom_params=custom_params,
                                                    parse_dates=["starttime","endtime"]
                                                    )
                except Exception as e:
                    logger.error(f"Error loading data from {db_path}: {e}")
                    continue
                
                if df.empty:
                    logger.warning(f"No dataframe found for {stat} in {db_path}")
                    continue
                
                # Set the DataFrame index and create a MultiIndex for columns
                df.set_index(['starttime', 'endtime'], inplace=True)
                stat_columns = [stat] * len(df.columns.tolist())
                multi_columns = list(zip(stat_columns, df.columns.tolist()))
                multi_columns = pd.MultiIndex.from_tuples(multi_columns)
                df.columns = multi_columns
                
                dfs_stats.append(df)
            
            if not dfs_stats:
                logger.error(f"No data recorded in {db_path}")
                continue
            
            # Concatenate dataframes for the current path
            df = pd.concat(dfs_stats, axis=1)
            
            # Store or append DataFrame to the collection
            if netstainpath not in all_dfs:
                all_dfs[netstainpath] = [df]
            else:
                all_dfs[netstainpath].append(df)
        
        # Process all collected DataFrames
        for key, sta_dfs in all_dfs.items():
            df = pd.concat(sta_dfs, axis=0)
            
            # Remove duplicated dates and clean up DataFrame
            df.reset_index(inplace=True)
            conversion = {('starttime', ''): "starttime",
                        ('endtime', ''): "endtime"}
            df = df.drop_duplicates(subset=list(conversion.keys()))
            df.set_index(list(conversion.keys()), inplace=True)
            df = df.rename_axis(index=conversion)
            all_dfs[key] = df
        
        if not all_dfs:
            logger.error("No data recorded")
            return None
        
        # Concatenate all DataFrames and sort by 'starttime'
        df = pd.concat(list(all_dfs.values()), axis=1)
        df = (df
            .sort_values(by='starttime')  # Sort the DataFrame by 'starttime'
            .reset_index()                # Reset the index and drop the old index
            .set_index(['starttime', 'endtime'])  # Set 'starttime' and 'endtime' as the new index
        )
        
        return df

               
def plot_rolling_stats(stats, freq, strid_list=[],
                       stat_type="availability",
                       starttime=None,
                       endtime=None,
                       colorbar=None,
                       major_step=7,
                       major_format = "%Y-%m-%d %H:%M:%S",
                       filter_stations=None,
                       show=True,
                       out=None):
    """
    Plots rolling statistics data as a heatmap with optional color bar and time axis customization.

    Args:
        stats (pd.DataFrame): DataFrame containing statistical data with a MultiIndex of 'starttime' and 'endtime'.
        freq (str): Frequency string for resampling, e.g., '7D' for 7-day intervals.
                    See here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        strid_list (list): List of specific identifiers to include in the plot. Defaults to empty list.
        stat_type (str): Type of statistic to plot. Defaults to "availability".
        starttime (pd.Timestamp or None): Start time for filtering the data. Defaults to None.
        endtime (pd.Timestamp or None): End time for filtering the data. Defaults to None.
        colorbar (StatsColorBar or None): Optional color bar configuration. Defaults to None.
        major_step (int): Interval between major ticks on the x-axis, specified in seconds. Defaults to 7.
        major_format (str, optional): Format string for major tick labels. Defaults to "%Y-%m-%d %H:%M:%S".
        filter_stations (list or None,optional): List of stations to remove 
        show (bool): Whether to display the plot. Defaults to True.
        out (str or None): File path to save the plot. Defaults to None.

    Returns:
        tuple: A tuple containing the figure, primary axis, and secondary axis objects.
    """
    
    # Extract column names and filter based on stat_type and strid_list
    stats_columns = stats.columns.to_list()
    right_columns = []
    for stat, strid in stats_columns:
        if stat_type != stat:
            continue
        if filter_stations is not None:
            net,sta,loc,cha = strid.split(".")
            if sta in filter_stations:
                continue
        if not strid_list or strid in strid_list:
            right_columns.append((stat, strid))
    
    if not right_columns:
        raise ValueError("No data to analyze.")
    else:
        # print(f"Data to analyze: {right_columns}")
        print(f"Compiling data...")

    # Filter and prepare the DataFrame
    stat = stats[right_columns]
    stat = stat.fillna(0)
    stat.columns = stat.columns.droplevel()

    # Filter based on provided starttime and endtime
    if starttime is not None:
        stat = stat.loc[stat.index.get_level_values('starttime') >= starttime]
        tmp_starttime_row0 = stat.index.get_level_values('starttime').min()
        tmp_endtime_row0 = stat.index.get_level_values('endtime').min()
        tmp_deltatime = tmp_endtime_row0 - tmp_starttime_row0
        # print(starttime+tmp_deltatime)
        # exit()
        
        if tmp_starttime_row0 != starttime:
            # Create a new row with NaN values for every column
            new_row = pd.DataFrame(index=pd.MultiIndex.from_tuples([(starttime, starttime+tmp_deltatime)], 
                                                                   names=['starttime', 'endtime']),
                                columns=stat.columns)

            # Append new row
            stat = pd.concat([stat, new_row])
            stat = stat.fillna(0)
    else:
        starttime = stat.index.get_level_values('starttime').min()

    if endtime is not None:
        stat = stat.loc[stat.index.get_level_values('endtime') <= endtime]
        tmp_endtime_lastrow = stat.index.get_level_values('endtime').max()
        tmp_starttime_lastrow  = stat.index.get_level_values('starttime').max()
        tmp_deltatime = tmp_endtime_lastrow - tmp_starttime_lastrow
        
        if tmp_endtime_lastrow != endtime:
            # Create a new row with NaN values for every column
            new_row = pd.DataFrame(index=pd.MultiIndex.from_tuples([(endtime-tmp_deltatime, endtime)], 
                                                                   names=['starttime', 'endtime']),
                                columns=stat.columns)

            # Append new row
            stat = pd.concat([stat, new_row])
            stat = stat.fillna(0)
    else:
        endtime = stat.index.get_level_values('endtime').max()

    # Sort by index levels
    stat = stat.sort_index(level=['starttime', 'endtime'])
    # Reset index and resample data to specified frequency
    stat = stat.reset_index(level='endtime', drop=True)
    stat = stat.resample(freq).mean()

    # Calculate endtime based on the frequency
    stat['endtime'] = stat.index + pd.to_timedelta(freq)

    # Set a new MultiIndex with 'starttime' and 'endtime'
    stat = stat.reset_index(level='starttime', drop=False)
    stat.set_index(['starttime', 'endtime'], inplace=True)

    # Prepare y-axis and x-axis information
    perc = stat_type == "availability"
    yaxis_info = ut.sort_yaxis_info(stat=stat, perc=perc)
    xaxis_info = ut.sort_xaxis_info(stat=stat, major_step=major_step,major_format=major_format)

    # Configure the color bar
    if colorbar is None:
        cbar_info = ut.StatsColorBar(stat_type).get_colorbar_info()
    else:
        cbar_info = colorbar.get_colorbar_info(
            cmap_name=colorbar.cmap_name,
            bad_colorname=colorbar.bad_colorname,
            zero_colorname=colorbar.zero_colorname
        )

    # Create the figure and axes
    fig = plt.figure(figsize=(12, 12))
    ax = host_subplot(111)
    ax1 = ax.twinx()

    # Set limits for both axes
    ax.set(xlim=(0, len(stat.index)), ylim=(0, len(stat.columns)))
    ax1.set(xlim=(0, len(stat.index)), ylim=(0, len(stat.columns)))

    # Plot the heatmap
    im = ax.pcolormesh(
        stat[yaxis_info["order"]].T.iloc[::-1],
        cmap=cbar_info["cmap"], alpha=1,
        norm=cbar_info["norm"]
    )

    # Configure x-axis and y-axis ticks and labels
    ax.set_yticks(np.arange(stat.shape[1])[::-1] + 0.5, minor=False)
    ax.set_yticks(np.arange(stat.shape[1])[::-1], minor=True)
    ax.set_xticks(range(len(xaxis_info["minor"])), minor=True)
    ax.set_xticks(range(0, len(xaxis_info["minor"]), major_step), minor=False)
    ax.set_yticklabels(yaxis_info["labels"], minor=False)
    ax.set_xticklabels(xaxis_info["minor"], minor=True)
    ax.set_xticklabels(xaxis_info["major"], minor=False)
    
    # Increase size of major ticks on x-axis
    ax.tick_params(axis='x', which='major', size=8)  

    # Hide minor ticks on the x-axis
    [t.label1.set_visible(False) for t in ax.xaxis.get_minor_ticks()]

    # Configure grid lines
    ax.grid(linestyle='--', zorder=12, which='minor')
    ax.grid(linestyle='-', linewidth=1.45, zorder=24, which='major', axis="x", color="black")
    ax.grid(linestyle='--', zorder=12, which='minor', axis="x")
    
    # Rotate x-axis date labels
    fig.autofmt_xdate()
    
    # Configure the secondary y-axis for availability data
    ax1.set_yticks(np.arange(stat.shape[1])[::-1] + 0.5, minor=False)
    ax1.set_yticks(yaxis_info["ticks"], minor=True)

    if yaxis_info["availability"]:
        ax1.set_yticklabels(yaxis_info["availability"], minor=False, fontdict={"fontsize": 8})
        ax1.set_ylabel("Average availability")
        pad = 0.1
    else:
        ax1.set_yticks([])
        ax1.yaxis.set_tick_params(labelleft=False, labelright=False)
        pad = 0.05

    ax1.grid(linestyle='-', linewidth=1.5, zorder=24, which='minor', axis="y", color="black")

    # Hide minor ticks on the x-axis of the secondary y-axis
    [t.label1.set_visible(False) for t in ax1.xaxis.get_minor_ticks()]

    # Add the color bar
    cbar = fig.colorbar(im, shrink=0.7, format=cbar_info["format"],
                        ticks=cbar_info["ticks"], pad=pad, ax=ax)
    cbar_name = stat_type.split()
    cbar_name[0] = cbar_name[0].capitalize()
    cbar_name = ''.join(cbar_name)
    cbar.set_label(f"{cbar_name}")
    cbar.ax.tick_params(size=0)

    # Adjust layout
    plt.tight_layout()

    # Save the figure if output path is provided
    if out is not None:
        if not os.path.isdir(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
        fig.savefig(out)

    # Show the plot if requested
    if show:
        plt.show()

    return fig, ax, ax1            
                
                
            
if __name__ == "__main__":   
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    import matplotlib.colors as mcolors
    
    starttime = UTCDateTime("2024-08-01T00:00:00")
    endtime = UTCDateTime("2024-08-24T00:00:00")
    
    # starttime = UTCDateTime("2024-01-01T00:00:00")
    # endtime = UTCDateTime("2024-08-01T00:00:00")
    wav_restrictions = WaveformRestrictions(
                "TX,2T,4T,4O",
                "*",
              "*","*",
              starttime,endtime,
              location_preferences=["", "00", "20", "10", "40"],
              instrument_preferences=["HH","","BH", "EH", "HN", "HL"],
              remove_networks=[], 
              remove_stations=[],
            #   filter_domain=[-104.6,-104.4,31.6,31.8], #lonw,lone,lats,latn #subregion
            #   filter_domain=[-104.5,-103.5,31,32], #lonw,lone,lats,latn #big region
            #   filter_domain=[-105,-103.5,31,32], #lonw,lone,lats,latn #AOI1
              filter_domain=[-104.84329,-103.79942,31.39610,31.91505], #lonw,lone,lats,latn #AOI2
              
              )   
    client= Client("TEXNET")
    provider = Provider(client=client,
                        wav_restrictions=wav_restrictions)
    
    # ### TO SCAN    
    # db_path = "/home/emmanuel/ecastillo/dev/delaware/data/metadata/delaware_database2024"
    # scanner = Scanner(db_path,providers=[provider],configure_logging=True)
    # scanner.scan(step=3600,wav_length=86400,level="station",n_processor=4)
    
    
    # ## plotting results
    # db_path = "/home/emmanuel/ecastillo/dev/delaware/data/metadata/delaware_database2024"
    # scanner = Scanner(db_path,providers=[provider],configure_logging=False)
    # stats =scanner.get_stats(network="4O",station="*",
    #                   location="*",instrument="[CH]H",
    #                   starttime=UTCDateTime("2024-01-01 00:00:00"),
    #                   endtime=UTCDateTime("2024-08-01 00:00:00"),
    #                 #   stats=["availability"]
    #                   )
    # print(stats)
    # min = 60
    # hour = 3600
    # day = 86400
    # # colorbar = ut.StatsColorBar(stat="availability",
    # #                             label_dict={"No gaps":[0,1e-5],
    # #                                         r"$\leq 1$ hour":[1e-5,hour],
    # #                                         r"$\leq 12$ hours":[hour,hour*12],
    # #                                         r"$\leq 1$ day":[hour*12,day],
    # #                                         r"$\geq 1$ day":[day,day+0.1],
    # #                                         }
    # #                             )
    # colorbar = ut.StatsColorBar(stat="availability",
    #                             # cmap_name='Greens',
    #                             cmap_name='YlGn',
    #                             bad_colorname="red",
    #                             label_dict={"[0,20]":[0,20],
    #                                         r"[20,40]":[20,40],
    #                                         r"[40,60]":[40,60],
    #                                         r"[60,80]":[60,80],
    #                                         r"[80,100]":[80,100],
    #                                         # r"100":[99.5,100],
    #                                         }
    #                             )
    # plot_rolling_stats(stats=stats,freq="7D",major_step=4,
    #                    colorbar=colorbar
    #                 #    starttime=UTCDateTime("2024-06-01 00:00:00").datetime
    #                    )
    
    ## plotting results 2
    db_path = "/home/emmanuel/ecastillo/dev/delaware/data/database/delaware_database*"
    scanner = Scanner(db_path,providers=[provider],configure_logging=False)
    stats =scanner.get_stats(network="*",station="*",
                      location="*",instrument="[CH]H",
                      starttime=UTCDateTime("2019-01-01 00:00:00"),
                      endtime=UTCDateTime("2024-08-01 00:00:00"),
                      
                      
                    #   stats=["availability"]
                      )
    min = 60
    hour = 3600
    day = 86400
    colorbar = ut.StatsColorBar(stat="availability",
                                # cmap_name='Greens',
                                cmap_name='YlGn',
                                bad_colorname="red",
                                label_dict={"[0,20)":[0,20],
                                            r"[20,40)":[20,40],
                                            r"[40,60)":[40,60],
                                            r"[60,80)":[60,80],
                                            r"[80,100]":[80,100],
                                            # r"[80,99)":[80,99],
                                            # r"[99-100]":[99,100],
                                            }
                                )
    plot_rolling_stats(stats=stats,freq="1MS",major_step=4,
                       colorbar=colorbar,
                       major_format="%Y-%m-%d",
                       filter_stations=["PB17"],
                    #    starttime=datetime.datetime.strptime("2019-01-01 00:00:00", 
                    #                                         "%Y-%m-%d %H:%M:%S"),
                    #    endtime=datetime.datetime.strptime("2025-01-01 00:00:00", 
                    #                                         "%Y-%m-%d %H:%M:%S")
                       )
    
    
    
    # i2q = provider.get_info_to_query(level="channel")
    # print(i2q)
    # print(provider.__str__(True))
    # print(provider)
    # scanner = Scanner()     