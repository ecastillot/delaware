# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-08-17 11:16:32
#  * @modify date 2024-08-17 11:16:32
#  * @desc [description]
#  */
import scan.utils as ut
from scan.stats import get_rolling_stats
import logging
import pandas as pd
import os
import glob
import datetime
import concurrent.futures as cf
from core.database import save_dataframe_to_sqlite,load_dataframe_from_sqlite

logger = logging.getLogger("delaware.scan.scanner")

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
                 
    def get_stats(self,network,station,location,instrument,
             starttime, endtime,stats=["availability","gaps_counts"]):
        
        db_name = ".".join((network,station,location,instrument))
        key = os.path.join(self.db_folder_path,db_name+"**")
        db_paths = glob.glob(key)
        
        if not db_paths:
            logger.info(f"No paths using this key in a glob search: {key} paths")
        
        logger.info(f"Loading: {len(db_paths)} paths")
        
        format = "%Y-%m-%d %H:%M:%S"
        all_dfs = []
        for i,db_path in enumerate(db_paths,1):
            logger.info(f"Loading: {i}/{len(db_paths)} {db_path}")
            dfs_stats = []
            for stat in stats:
                starttime_str = starttime.strftime(format)
                endtime_str = endtime.strftime(format)
                
                try:
                    df = load_dataframe_from_sqlite(db_name=db_path,
                                                    table_name=stat,
                                                    starttime=starttime_str,
                                                    endtime=endtime_str
                                                    )
                except Exception as e:
                    logger.error(e)
                    continue
                
                df.set_index(['starttime', 'endtime'], inplace=True)

                stat_columns = [stat]*len(df.columns.tolist())
                multi_columns = list(zip(stat_columns,df.columns.tolist()))
                multi_columns = pd.MultiIndex.from_tuples(
                                multi_columns,
                                    )

                # Assign MultiIndex to DataFrame columns
                df.columns = multi_columns
                
                dfs_stats.append(df)
            
            if not dfs_stats:
                logger.error(f"No data recorded in {db_path}")
                continue
            
            df = pd.concat(dfs_stats,axis=1)
            all_dfs.append(df)
        
        if not all_dfs:
            logger.error(f"No data recorded")
            return None
            
        df = pd.concat(all_dfs,axis=1)
        df = (df
            .sort_values(by='starttime')  # Sort the DataFrame by 'starttime'
            .reset_index()       # Reset the index and drop the old index
            .set_index(['starttime', 'endtime'])  # Set 'starttime' and 'endtime' as the new index
            )
        
        return df
                
def plot_rolling_stats(stats,freq,strid_list=[],
                       stat_type="availability",
                       starttime = None,
                       endtime = None,
                       colorbar = None,
                       major_step=7,
                    #    plot_availability=True,
                       show=True,
                       out = None):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import host_subplot
    
    stats_columns =  stats.columns.to_list()
    
    right_columns = []
    for stat,strid in stats_columns:
        if stat_type != stat:
            continue
        if not strid_list:
            right_columns.append((stat,strid))
        else:
            if strid in strid_list:
                right_columns.append((stat,strid))
                
    if not right_columns:
        raise ValueError("No data to analize")
    else:
        print(f"Data to analyze: {right_columns}")
        

    stat = stats[right_columns]
    stat =stat.fillna(0)
    stat.columns = stat.columns.droplevel()
    
    # Filter based on index values 
    if starttime != None:
        stat = stat.loc[
            (stat.index.get_level_values('starttime') >= starttime) ]
    else:
        starttime = stat.index.get_level_values('starttime').min()
    if endtime != None:
        stat = stat.loc[
            (stat.index.get_level_values('endtime') <= endtime)]
    else:
        endtime = stat.index.get_level_values('endtime').max()
    
    # Reset index to make 'starttime' a column
    stat = stat.reset_index(level='endtime',drop=True)
    # Resample the data over 7-day bins and calculate the mean
    stat =stat.resample(freq).mean()
    
    # Calculate endtime by adding the frequency to starttime
    stat['endtime'] = stat.index + pd.to_timedelta(freq)

    # Set a new MultiIndex with starttime and endtime
    stat = stat.reset_index(level='starttime',drop=False)
    stat.set_index(['starttime', 'endtime'], inplace=True)
    
    
    if stat_type == "availability":
        perc = True
        
    yaxis_info = ut.sort_yaxis_info(stat=stat,perc=perc)
    xaxis_info = ut.sort_xaxis_info(stat=stat,major_step=major_step)
    
    
    if colorbar is None:
        cbar_info = ut.StatsColorBar(stat_type).get_colorbar_info()
    else:
        cbar_info = colorbar.get_colorbar_info(cmap_name=colorbar.cmap_name,
                                               bad_colorname=colorbar.bad_colorname,
                                               zero_colorname=colorbar.zero_colorname)
        
    # print(cbar_info)
        
    # print(range(0,len(xaxis_info["minor"]),major_step))
    # exit()
    # exit()


    fig = plt.figure(figsize=(12,12))
    ax = host_subplot(111)
    # ax.set_facecolor('lightgray')
    ax1 = ax.twinx()

    ax1.set(xlim=(0, len(stat.index)),ylim=(0, len(stat.columns)))
    ax.set(xlim=(0, len(stat.index)),ylim=(0, len(stat.columns)))

    print(stat[yaxis_info["order"]].T.iloc[::-1])
    # print(stat[yaxis_info["order"]].T.info())
    im = ax.pcolormesh(stat[yaxis_info["order"]].T.iloc[::-1], 
                       cmap=cbar_info["cmap"], alpha=1,
                       norm = cbar_info["norm"])
    
    # plt.show()
    # exit()

    ax.set_yticks(np.arange(stat.shape[1])[::-1] + 0.5, minor=False)
    ax.set_yticks(np.arange(stat.shape[1])[::-1] , minor=True)

    ax.set_xticks(range(0,len(xaxis_info["minor"])), minor=True)
    ax.set_xticks(range(0,len(xaxis_info["minor"]),major_step), minor=False)


    ax.set_yticklabels(yaxis_info["labels"],minor=False)
    # print(np.arange(stat.shape[1]) + 0.5)
    # print(yaxis_info["labels"])
    # plt.show()
    # exit() 
    ax.set_xticklabels(xaxis_info["minor"], minor=True)
    ax.set_xticklabels(xaxis_info["major"], minor=False)

    #minor ticks false
    [t.label1.set_visible(False) for t in ax.xaxis.get_minor_ticks()]

    plt.tick_params(left = False)

    ax.grid(linestyle='--',zorder=12,which='minor')
    ax.grid(linestyle='-',linewidth=1.45,zorder=24,which='major',axis="x",color="black")
    ax.grid(linestyle='--',zorder=12,which='minor',axis="x")

    ## Rotate date labels automatically
    fig.autofmt_xdate()

    
    # ax1.set_yticks(np.arange(stat.shape[1])[::-1] + 0.5, minor=False)
    ax1.set_yticks(np.arange(stat.shape[1])[::-1] + 0.5, minor=False)
    ax1.set_yticks(yaxis_info["ticks"] , minor=True)

    # print(yaxis_info)
    # exit()

    if yaxis_info["availability"]:
        ax1.set_yticklabels(yaxis_info["availability"],minor=False,
                            fontdict={"fontsize":8})
        ax1.set_ylabel("Average availability")
        pad = 0.2
    else:
        ax1.set_yticks([])
        ax1.yaxis.set_tick_params(labelleft=False,labelright=False)
        pad = 0.1

    ax1.grid(linestyle='-',linewidth=1.5,zorder=24,which='minor',axis="y",color="black")
    
    #minor ticks false
    [t.label1.set_visible(False) for t in ax1.xaxis.get_minor_ticks()]


    cbar = fig.colorbar(im,shrink=0.7,format=cbar_info["format"],
                        ticks=cbar_info["ticks"], pad=pad,ax=ax)
    cbar.set_label(f"{stat_type}")
    cbar.ax.tick_params(size=0)


    plt.tight_layout()

    if out != None:
        if not os.path.isdir(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
        fig.savefig(out)
        # fig.savefig(out,bbox_inches='tight')
    if show:
        plt.show()
    return fig,ax,ax1                       
                
                
            
if __name__ == "__main__":   
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    import matplotlib.colors as mcolors
    starttime = UTCDateTime("2024-07-22T23:00:00")
    endtime = UTCDateTime("2024-08-01T00:00:00")
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
              filter_domain=[-104.5,-103.5,31,32], #lonw,lone,lats,latn #big region
              )   
    client= Client("TEXNET")
    # print(client.__dict__)
    provider = Provider(client=client,
                        wav_restrictions=wav_restrictions)
    
    db_path = "/home/emmanuel/ecastillo/dev/delaware/data/metadata/delaware_database"
    # scanner = Scanner(db_path,providers=[provider])
    scanner = Scanner(db_path,providers=[provider],configure_logging=False)
    
    # scanner.scan(step=3600,wav_length=86400,level="station",n_processor=4)
    
    stats =scanner.get_stats(network="4O",station="*",
                      location="*",instrument="HH?",
                      starttime=UTCDateTime("2024-01-01 00:00:00"),
                      endtime=UTCDateTime("2024-08-01 00:00:00"),
                    #   stats=["availability"]
                      )
    print(stats)
    
    min = 60
    hour = 3600
    day = 86400
    # colorbar = ut.StatsColorBar(stat="availability",
    #                             label_dict={"No gaps":[0,1e-5],
    #                                         r"$\leq 1$ hour":[1e-5,hour],
    #                                         r"$\leq 12$ hours":[hour,hour*12],
    #                                         r"$\leq 1$ day":[hour*12,day],
    #                                         r"$\geq 1$ day":[day,day+0.1],
    #                                         }
    #                             )
    # cmap = mcolors.LinearSegmentedColormap.from_list('red_to_green', ['red', 'green'])
    colorbar = ut.StatsColorBar(stat="availability",
                                # cmap_name='Greens',
                                cmap_name='YlGn',
                                bad_colorname="red",
                                label_dict={"[0,20]":[0,20],
                                            r"[20,40]":[20,40],
                                            r"[40,60]":[40,60],
                                            r"[60,80]":[60,80],
                                            r"[80,100]":[80,100],
                                            # r"100":[99.5,100],
                                            }
                                )
    plot_rolling_stats(stats=stats,freq="7D",major_step=4,
                       colorbar=colorbar
                    #    starttime=UTCDateTime("2024-06-01 00:00:00").datetime
                       )
    
    
    
    # i2q = provider.get_info_to_query(level="channel")
    # print(i2q)
    # print(provider.__str__(True))
    # print(provider)
    # scanner = Scanner()     