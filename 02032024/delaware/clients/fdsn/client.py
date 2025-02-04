
import os
import pandas as pd
from .utils import get_stations_info, get_custom_info, save_info
from utdquake.tools.stats import get_rolling_stats
from obspy.clients.fdsn import Client as FDSNClient

class Client(FDSNClient):
    """
    A client class for retrieving and calculating rolling statistics on seismic data.

    Inherits from:
        FDSNClient: Base class for FDSN web service clients.

    Attributes:
        output (str): Path to the SQLite database file for saving results.
        step (int): Step size for the rolling window in seconds.
    """

    def __init__(self,*args, **kwargs):
        """
        Initializes the Client class by calling the constructor 
        of the base FDSN Client class.

        Parameters:
        *args : variable length argument list
            Positional arguments passed to the base class constructor.
        **kwargs : variable length keyword arguments
            Keyword arguments passed to the base class constructor.
        """
        super().__init__(*args, **kwargs)

    def __get_custom_event_ids(self, starttime, endtime, ev_kwargs):
        """
        Retrieve custom event IDs from a catalog of seismic events.

        Parameters:
            starttime (UTCDateTime): Start time for the event search.
            endtime (UTCDateTime): End time for the event search.
            ev_kwargs (dict): Additional keyword arguments for event filtering.

        Returns:
            list: A list of custom event IDs.
        """

        # Retrieve the catalog of events using the get_events method
        catalog = self.get_events(starttime, endtime, **ev_kwargs)

        # Initialize an empty list to store event IDs
        ev_ids = []

        # Mode to determine the format of the event ID, initialized as None
        mode = None

        # Iterate through each event in the catalog
        for event in catalog:
            
            # Extract additional data from the event
            extra_data_src = event.extra.datasource.value
            extra_ev_id = event.extra.eventid.value

            # Define potential event ID formats
            potential_ev_ids = {
                "1": extra_data_src + extra_ev_id,
                "2": extra_ev_id
            }

            # Determine the mode (event ID format) if not already set
            if mode is None:
                for p_mode, p_ev_id in potential_ev_ids.items():
                    try:
                        # Test if the event ID exists in the catalog
                        self.get_events(starttime, endtime, eventid=p_ev_id)
                        mode = p_mode
                        break  # Exit the loop once a valid mode is found
                    except Exception:
                        pass

            # Raise an exception if no valid event ID format is found
            if mode is None:
                raise Exception(f"No event found using any of: {potential_ev_ids}")

            # Use the determined mode to select the correct event ID
            ev_id = potential_ev_ids[mode]
            ev_ids.append(ev_id)

        # Return the list of event IDs
        return ev_ids

    def get_custom_stations(self, output_folder=None, **sta_kwargs):
        """
        Retrieve custom station information and optionally save it to a CSV file.

        Args:
            output_folder (str, optional): Path to the folder where the station 
                information will be saved. If None, the information will not be saved.
            **sta_kwargs: Additional keyword arguments to filter stations 
                when calling `self.get_stations`.

        Returns:
            pandas.DataFrame: A DataFrame containing the station information, including:
                - "sta_id": Network.Station.
                - "network": Network code to which the station belongs.
                - "station": Station code.
                - "latitude": Latitude of the station.
                - "longitude": Longitude of the station.
                - "elevation": Elevation of the station.
                - "starttime": Start date and time of the station's operation.
                - "endtime": End date and time of the station's operation.
                - "site_name": Name of the site.
        """
        # Retrieve the inventory using the provided keyword arguments
        inv = self.get_stations(**sta_kwargs)

        # Extract station information into a DataFrame
        sta_info = get_stations_info(inv)

        # If an output folder is specified, save the DataFrame as a CSV file
        if output_folder is not None:
            # Create the output folder if it doesn't exist
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)

            # Define the full path for the CSV file
            path = os.path.join(output_folder, "stations.csv")

            # Save the station information to the CSV file
            sta_info.to_csv(
                path,
                mode='a',  # Append mode
                header=not pd.io.common.file_exists(path),  # Add header only if the file doesn't exist
                index=False  # Do not write row numbers
            )

        return sta_info

    def get_custom_events(self, starttime, endtime,  max_events_in_ram=1e6,
                      output_folder=None, drop_level=True, debug=False, **ev_kwargs):
        """
        Retrieves custom seismic event data, including origins, picks, and magnitudes.

        Parameters:
        ----------
        starttime : obspy.core.utcdatetime.UTCDateTime
            Limit results to time series samples on or after the specified start time.
        endtime : obspy.core.utcdatetime.UTCDateTime
            Limit results to time series samples on or before the specified end time.
        max_events_in_ram : int, optional, default=1e6
            Maximum number of events to hold in memory (RAM) before stopping or 
            prompting to save the data to disk.
        output_folder : str, optional, default=None
            Folder path where the event data will be saved if provided. If not 
            specified, data will only be stored in memory.
        drop_level : bool, optional, default=True
            If True, the origin DataFrame will have only one hierarchical level.
        debug: bool, optional, default = False
            Print the events it is trying to get.
        **ev_kwargs : variable length keyword arguments
            Additional arguments passed to the `get_events` method.

        Returns:
        -------
        tuple
            A tuple containing:
            - pd.DataFrame: Origins for all events.
            - pd.DataFrame: Picks for all events.
            - pd.DataFrame: Magnitudes for all events.
        """
        
        # # Retrieve the catalog of events using the get_events method
        ev_ids = self.__get_custom_event_ids(starttime, endtime,ev_kwargs)
        
        # Initialize lists to store origins, picks, and magnitudes
        all_origins, all_picks, all_mags = [], [], []

        # Loop through each event ID to gather detailed event information
        for k,ev_id in enumerate(ev_ids[::-1],1):
            
            # Print debug
            if debug:
                print(f"Event id {k}/{len(ev_ids[::-1])}: {ev_id}")
            
            # Catalog with arrivals. This is a workaround to retrieve 
            # arrivals by specifying the event ID.
            cat = self.get_events(eventid=ev_id,**ev_kwargs)

            # Get the first event from the catalog
            event = cat[0]

            # Extract custom information for the event
            origin, picks, mags = get_custom_info(ev_id, event, drop_level)

            info = {
                "origin": origin,
                "picks": picks,
                "mags": mags
            }

            # Save information to the output folder, if specified
            if output_folder is not None:
                if not os.path.isdir(output_folder):
                    os.makedirs(output_folder)
                save_info(output_folder, info=info)

            # Append information to the lists or break if memory limit is reached
            if len(all_origins) < max_events_in_ram:
                all_origins.append(origin)
                all_picks.append(picks)
                all_mags.append(mags)
            else:
                if output_folder is not None:
                    print(f"max_events_in_ram: {max_events_in_ram} is reached. "
                        "But it is still saving on disk.")
                else:
                    print(f"max_events_in_ram: {max_events_in_ram} is reached. "
                        "It is recommended to save the data on disk using the 'output_folder' parameter.")
                    break

        # Concatenate data from all events, if multiple events are found
        if len(ev_ids) > 1:
            all_origins = pd.concat(all_origins, axis=0)
            all_picks = pd.concat(all_picks, axis=0)
            all_mags = pd.concat(all_mags, axis=0)
        else:
            # If only one event is found, retain the single DataFrame
            all_origins = all_origins[0]
            all_picks = all_picks[0]
            all_mags = all_mags[0]

        return all_origins, all_picks, all_mags

    def get_stats(self, step, network, station, location, channel, starttime, endtime, output=None, **kwargs):
        """
        Retrieve waveforms and compute rolling statistics for the specified time interval.

        Parameters:
        ----------
        step : int
            Step size for the rolling window in seconds.
        network : str
            Select one or more network codes. These can be SEED network
            codes or data center-defined codes. Multiple codes can be
            comma-separated (e.g., "IU,TA"). Wildcards are allowed.
        station : str
            Select one or more SEED station codes. Multiple codes
            can be comma-separated (e.g., "ANMO,PFO"). Wildcards are allowed.
        location : str
            Select one or more SEED location identifiers. Multiple
            identifiers can be comma-separated (e.g., "00,01"). Wildcards are allowed.
        channel : str
            Select one or more SEED channel codes. Multiple codes
            can be comma-separated (e.g., "BHZ,HHZ").
        starttime : obspy.core.utcdatetime.UTCDateTime
            Limit results to time series samples on or after the
            specified start time.
        endtime : obspy.core.utcdatetime.UTCDateTime
            Limit results to time series samples on or before the
            specified end time.
        output : str, optional
            Path to the SQLite database file for saving results. Defaults to None.
        **kwargs : dict
            Additional keyword arguments passed to the `self.get_waveforms` method.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing rolling statistics for each interval, including:
            - Availability percentage
            - Gaps duration
            - Overlaps duration
            - Gaps count
            - Overlaps count
        """
        # Retrieve waveforms using the get_waveforms method
        st = self.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
            **kwargs
        )

        # Compute rolling statistics for the retrieved waveforms
        stats = get_rolling_stats(
            st=st,
            step=step,
            starttime=starttime.datetime,
            endtime=endtime.datetime,
            sqlite_output=output
        )

        return stats