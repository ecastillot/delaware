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

    def scan(self, step, wav_length=86400, level="station", n_processor=1):
        """
        Scan the waveform data for each provider and save results to the database.

        Parameters:
        -----------
        step : int
            The step size for rolling statistics calculation.
        wav_length : int, optional
            Length of each waveform chunk in seconds. Defaults to 86400 seconds (1 day).
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
                        logger.error(f"{info}|{chunk_starttime}-{chunk_endtime}"+f"\n{e}")
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
                
                        
                
                
            
if __name__ == "__main__":   
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    starttime = UTCDateTime("2024-04-18T23:00:00")
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
    scanner = Scanner(db_path,providers=[provider])
    
    scanner.scan(step=3600,wav_length=86400,level="station",n_processor=4)
    
    # stats =scanner.get_stats(network="TX",station="PB*",
    #                   location="*",instrument="HH?",
    #                   starttime=UTCDateTime("2024-01-01 00:00:00"),
    #                   endtime=UTCDateTime("2024-08-01 00:00:00"),
    #                 #   stats=["availability"]
    #                   )
    # print(stats)
    
    
    
    # i2q = provider.get_info_to_query(level="channel")
    # print(i2q)
    # print(provider.__str__(True))
    # print(provider)
    # scanner = Scanner()     