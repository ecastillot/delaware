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
                 filter_domain=[-180, 180, -90, 90]):
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
                   f"\n\tfilter_domain: {self.filter_domain}")
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
        if configure_logging:
            # Set up logging configuration
            logger.setLevel(logging.DEBUG)
            logger.propagate = 0  # Prevent log messages from being passed to higher loggers.

            # Console log handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # Formatter for log messages
            formatter = logging.Formatter(
                "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s")
            ch.setFormatter(formatter)

            # Add the handler to the logger
            logger.addHandler(ch)

        self.db_folder_path = db_folder_path
        self.providers = providers

    def scan(self, step, wav_length=86400, level="station"):
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
        """
        for provider in self.providers:
            logger.info(f"Provider {provider}")

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
                    logger.info(f"Loading the stream: {info}")

                    try:
                        # Fetch waveform data from the client
                        st = client.get_waveforms(network=net,
                                                  station=sta,
                                                  location=loc,
                                                  channel=cha,
                                                  starttime=chunk_starttime,
                                                  endtime=chunk_endtime)
                    except Exception as e:
                        logger.error(e)
                        st = False

                    if not st:
                        logger.error(f"No stream: {info}")
                        return

                    # Process the stream to standardize channels
                    st = ut.process_stream_common_channels(st,
                                                           location_preferences=wr.location_preferences,
                                                           instrument_preferences=wr.instrument_preferences)

                    logger.info(f"Scanning the stream: {info}")

                    # Compute and save rolling statistics
                    get_rolling_stats(st, step=step,
                                      starttime=chunk_starttime.datetime,
                                      endtime=chunk_endtime.datetime,
                                      sqlite_output=self.db_folder_path)

                # Perform the query for each set of parameters
                for info in i2q:
                    scan_query(info)   
                
                
                
            
if __name__ == "__main__":   
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    starttime = UTCDateTime("2024-01-01T00:00:00")
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
              filter_domain=[-104.6,-104.4,31.6,31.8] #lonw,lone,lats,latn
              )   
    client= Client("TEXNET")
    # print(client.__dict__)
    provider = Provider(client=client,
                        wav_restrictions=wav_restrictions)
    
    db_path = "/home/emmanuel/ecastillo/dev/delaware/data/metadata/delaware_database"
    scanner = Scanner(db_path,providers=[provider])
    scanner.scan(step=3600,wav_length=86400,level="station")
    # i2q = provider.get_info_to_query(level="channel")
    # print(i2q)
    # print(provider.__str__(True))
    # print(provider)
    # scanner = Scanner()     