# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-09-22 22:07:55
#  * @modify date 2024-09-22 22:07:55
#  * @desc [description]
#  */



import pandas as pd
import os
import sqlite3

# import objects
import os 
import glob
import warnings
import tqdm
import numpy as np
import pandas as pd
from datetime import timedelta
from obspy.core.util.misc import BAND_CODE
from obspy.clients.filesystem.sds import Client
from obspy.core.stream import _headonly_warning_msg
from obspy import UTCDateTime
from tqdm import tqdm
import concurrent.futures as cf
from obspy.clients.fdsn import Client as FDSNClient
from delaware.scan.stats import get_rolling_stats

class StatsClient(FDSNClient):
    """
    A client class for retrieving and calculating rolling statistics on seismic data.

    Inherits from:
        FDSNClient: Base class for FDSN web service clients.

    Attributes:
        output (str): Path to the SQLite database file for saving results.
        step (int): Step size for the rolling window in seconds.
    """

    def __init__(self, output, step, *args, **kwargs):
        """
        Initialize the StatsClient with output path, step size, and additional arguments.

        Args:
            output (str): Path to the SQLite database file for saving results.
            step (int): Step size for the rolling window in seconds.
            *args: Variable length argument list for additional parameters.
            **kwargs: Keyword arguments for the base class constructor.
        """
        self.output = output
        self.step = step
        super().__init__(*args, **kwargs)

    def get_stats(self, **kwargs):
        """
        Retrieve waveforms and compute rolling statistics for the specified time interval.

        Args:
            **kwargs: Keyword arguments for retrieving waveforms, including:
                - starttime (UTCDateTime): Start time of the data.
                - endtime (UTCDateTime): End time of the data.
                - Additional arguments required by `self.get_waveforms`.

        Returns:
            pd.DataFrame: A DataFrame containing rolling statistics for each interval, with columns including:
            - Availability percentage
            - Gaps duration
            - Overlaps duration
            - Gaps count
            - Overlaps count
        """
        # Extract start and end times from keyword arguments
        starttime = kwargs["starttime"]
        endtime = kwargs["endtime"]

        # Retrieve waveforms using base class method
        st = self.get_waveforms(**kwargs)

        # Compute rolling statistics for the retrieved waveforms
        stats = get_rolling_stats(
            st=st,
            step=self.step,
            starttime=starttime.datetime,
            endtime=endtime.datetime,
            sqlite_output=self.output
        )

        return stats

class LocalClient(Client):

    def __init__(self,root,fmt,**kwargs):
        """
        This script is an example to make a client class
        for specific data structure archive on local filesystem. 

        The mandatory parameters for LocalClient class is: root_path and field_name
        Example:
        ---------
        root = "/home/emmanuel/myarchive"
        fmt = "{year}-{month:02d}/{year}-{month:02d}-{day:02d}/{network}.{station}.{location}.{channel}.{year}.{julday:03d}"
        client = LocalClient(root,fmt)
        st = client.get_waveforms("YY","XXXX","00",
                                channel="HHZ",starttime = UTCDateTime("20220102T000100"),
                                endtime = UTCDateTime("20220102T000200"))
        
        Parameters:
        -----------
        root: str
            Path where is located the Local structure
        fmt: str
            The parameter should name the corresponding keys of the stats object, e.g. 
            "{year}-{month:02d}/{year}-{month:02d}-{day:02d}/{network}.{station}.{location}.{channel}.{year}.{julday:03d}"

        **kwargs SDS client additional args
        """
        self.root = root
        self.fmt = fmt
        super().__init__(root,**kwargs)

    def _get_filenames(self, network, station, location, channel, starttime,
                       endtime, sds_type=None):
        """
        Get list of filenames for certain waveform and time span.
        :type network: str
        :param network: Network code of requested data (e.g. "IU").
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
        :type location: str
        :param location: Location code of requested data (e.g. "").
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Time of interest.
        :type sds_type: str
        :param sds_type: None
        :rtype: str
        """
        sds_type = sds_type or self.sds_type
        # SDS has data sometimes in adjacent days, so also try to read the
        # requested data from those files. Usually this is only a few seconds
        # of data after midnight, but for now we play safe here to catch all
        # requested data (and with MiniSEED - the usual SDS file format - we
        # can use starttime/endtime kwargs anyway to read only desired parts).
        year_doy = set()
        # determine how far before starttime/after endtime we should check
        # other dayfiles for the data
        t_buffer = self.fileborder_samples / BAND_CODE.get(channel[:1], 20.0)
        t_buffer = max(t_buffer, self.fileborder_seconds)
        t = starttime - t_buffer
        t_max = endtime + t_buffer
        # make a list of year/doy combinations that covers the whole requested
        # time window (plus day before and day after)
        while t < t_max:
            year_doy.add((t.year,t.month,t.day, t.julday))
            t += timedelta(days=1)
        year_doy.add((t_max.year,t_max.month,t_max.day, t_max.julday))

        full_paths = set()
        for year,month,day,doy in year_doy:
            filename = self.fmt.format(
                            network=network, station=station, location=location,
                            channel=channel, year=year, month=month, 
                            day=day, julday=doy,sds_type=sds_type)
            full_path = os.path.join(self.sds_root, filename)
            full_paths = full_paths.union(glob.glob(full_path))
        
        return full_paths

    def _get_filename(self, network, station, location, channel, time, sds_type=None):
        """
        Get filename for certain waveform.
        :type network: str
        :param network: Network code of requested data (e.g. "IU").
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
        :type location: str
        :param location: Location code of requested data (e.g. "").
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Time of interest.
        """
        sds_type = sds_type or self.sds_type
        filename = self.fmt.format(
                    network=network, station=station, location=location,
                    channel=channel, year=time.year, month=time.month, 
                    day=time.day, doy=time.julday,sds_type=sds_type)
        return os.path.join(self.sds_root, filename)


### CustomClient

def get_custom_picks(event):
    """
    Extract custom picks information from an event.

    Parameters:
    event : Event object
        Seismic event from which to extract pick data.

    Returns:
    dict
        Dictionary with picks information, including network, station, and 
        phase details.
    """
    picks = {}
    
    # Loop through each pick in the event
    for pick in event.picks:
        picks[pick.resource_id.id] = {
            "network": pick.waveform_id.network_code if pick.waveform_id is not None else None,
            "station": pick.waveform_id.station_code if pick.waveform_id is not None else None,
            "location": pick.waveform_id.location_code if pick.waveform_id is not None else None,
            "channel": pick.waveform_id.channel_code if pick.waveform_id is not None else None,
            "phase_hint": pick.phase_hint,
            "arrival_time": pick.time.datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "time_lower_error": pick.time_errors.lower_uncertainty if pick.time_errors is not None else None,
            "time_upper_error": pick.time_errors.upper_uncertainty if pick.time_errors is not None else None,
            "author": pick.creation_info.author if pick.creation_info is not None else None,
            "filter_id": pick.filter_id.id if pick.filter_id is not None else None ,
            "method_id": pick.method_id.id if pick.method_id is not None else None,
            "polarity": pick.polarity,
            "evaluation_mode": pick.evaluation_mode,
            "evaluation_status": pick.evaluation_status
        }
    
    return picks

def get_custom_station_magnitudes(event):
    """
    Extract custom station magnitude information from an event.

    Parameters:
    event : Event object
        Seismic event from which to extract station magnitude data.

    Returns:
    dict
        Dictionary containing station magnitudes, including network and station details.
    """
    sta_mags = {}
    
    # Loop through each station magnitude in the event
    for sta_mag in event.station_magnitudes:
        sta_mags[sta_mag.resource_id.id] = {
            "network_code": sta_mag.waveform_id.network_code if sta_mag.waveform_id is not None else None,
            "station_code": sta_mag.waveform_id.station_code if sta_mag.waveform_id is not None else None,
            "location_code": sta_mag.waveform_id.location_code if sta_mag.waveform_id is not None else None,
            "channel_code": sta_mag.waveform_id.channel_code if sta_mag.waveform_id is not None else None,
            "mag": sta_mag.mag,
            "mag_type": sta_mag.station_magnitude_type
        }
    
    return sta_mags

def get_custom_arrivals(event):
    """
    Extract custom arrival information from an event and associate it with picks.

    Parameters:
    event : Event object
        Seismic event from which to extract arrival and pick data.

    Returns:
    tuple
        A tuple containing origin quality information and a DataFrame of 
        arrival contributions with associated picks.
    """
    ev_id = event.resource_id.id.split("/")[-1]
    origin = event.preferred_origin()
    
    # Get origin quality information
    info = dict(origin.quality)
    
    # Retrieve custom picks
    picks = get_custom_picks(event)
    
    arr_contributions = {}
    
    # Loop through each arrival in the origin
    for arrival in origin.arrivals:
        
        try:
            pick_info = picks[arrival.pick_id.id]
        except Exception as e:
            print(f"Event: {ev_id} | Pick not found:",e)
            continue
        
        
        pick_info["time_correction"] = arrival.time_correction
        pick_info["azimuth"] = arrival.azimuth
        pick_info["distance"] = arrival.distance
        pick_info["takeoff_angle"] = arrival.takeoff_angle
        pick_info["time_residual"] = arrival.time_residual
        pick_info["time_weight"] = arrival.time_weight
        pick_info["used"] = True
        
        arr_contributions[arrival.pick_id.id] = pick_info
    
    # Identify picks not used in arrivals
    not_used_ids = list(set(picks.keys()) - set(arr_contributions.keys()))
    
    for not_used_id in not_used_ids:
        pick_info = picks[not_used_id]
        pick_info["time_correction"] = None
        pick_info["azimuth"] = None
        pick_info["distance"] = None
        pick_info["takeoff_angle"] = None
        pick_info["time_residual"] = None
        pick_info["time_weight"] = None
        pick_info["used"] = False
        arr_contributions[not_used_id] = pick_info
    
    # Convert contributions to a DataFrame and drop duplicates
    arr_contributions = pd.DataFrame(list(arr_contributions.values()))
    arr_contributions = arr_contributions.drop_duplicates(ignore_index=True)
    arr_contributions.insert(0, "ev_id", event.resource_id.id.split("/")[-1])
    
    return info, arr_contributions

def get_custom_pref_mag(event):
    """
    Extract custom preferred magnitude information from an event.

    Parameters:
    event : Event object
        Seismic event from which to extract preferred magnitude data.

    Returns:
    tuple
        A tuple containing preferred magnitude information and a DataFrame of 
        station magnitude contributions.
    """
    ev_id = event.resource_id.id.split("/")[-1]
    magnitude = event.preferred_magnitude()
    
    # Get preferred magnitude information
    info = {
        "magnitude": magnitude.mag,
        "uncertainty": magnitude.mag_errors.uncertainty if magnitude.mag_errors is not None else None,
        "type": magnitude.magnitude_type,
        "method_id": magnitude.method_id.id.split("/")[-1] if magnitude.method_id is not None else None,
        "station_count": magnitude.station_count,
        "evaluation_status": magnitude.evaluation_status,
    }
    
    # Retrieve station magnitudes
    sta_mags = get_custom_station_magnitudes(event)
    
    mag_contributions = {}
    
    # Loop through each station magnitude contribution
    for used_sta_mag in magnitude.station_magnitude_contributions:
        
        try:
            sta_info = sta_mags[used_sta_mag.station_magnitude_id.id]
        except Exception as e:
            print(f"Event: {ev_id} | StationMagnitude not found:",e)
            continue
            
        sta_info["residual"] = used_sta_mag.residual
        sta_info["weight"] = used_sta_mag.weight
        sta_info["used"] = True
        mag_contributions[used_sta_mag.station_magnitude_id.id] = sta_info
    
    # Identify station magnitudes not used in contributions
    not_used_ids = list(set(sta_mags.keys()) - set(mag_contributions.keys()))
    
    for not_used_id in not_used_ids:
        sta_info = sta_mags[not_used_id]
        sta_info["residual"] = None
        sta_info["weight"] = None
        sta_info["used"] = False
        mag_contributions[not_used_id] = sta_info
    
    # Convert contributions to a DataFrame and drop duplicates
    mag_contributions = pd.DataFrame(list(mag_contributions.values()))
    mag_contributions = mag_contributions.drop_duplicates(ignore_index=True)
    mag_contributions.insert(0, "ev_id", event.resource_id.id.split("/")[-1])
    # mag_contributions.insert(0, "magnitude_id", magnitude.resource_id.id)
    
    return info, mag_contributions
    
def get_custom_origin(event):
    """
    Extract custom origin information from a seismic event.

    Parameters:
    event : Event object
        The seismic event from which to extract origin data.

    Returns:
    dict
        A dictionary containing event and origin information with 
        multilevel column structure.
    """
    # Get the preferred origin of the event
    origin = event.preferred_origin()
    
    # Prepare event information
    ev_info = {
        ("event", "ev_id"): event.resource_id.id.split("/")[-1] if event.resource_id is not None else None,
        ("event", "type"): event.event_type,
        ("event", "type_certainty"): event.event_type_certainty,
    }
    
    # Prepare location information
    loc_info = {
        ("origin_loc", "agency"): origin.creation_info.agency_id,
        ("origin_loc", "evaluation_mode"): origin.evaluation_mode,
        ("origin_loc", "evaluation_status"): origin.evaluation_status,
        ("origin_loc", "origin_time"): origin.time.datetime.strftime("%Y-%m-%d %H:%M:%S.%f") if origin.time is not None else None,
        ("origin_loc", "longitude"): origin.longitude,
        ("origin_loc", "longitude_error"): origin.longitude_errors.uncertainty,
        ("origin_loc", "latitude"): origin.latitude,
        ("origin_loc", "latitude_error"): origin.latitude_errors.uncertainty,
        ("origin_loc", "depth"): origin.depth,
        ("origin_loc", "depth_error"): origin.depth_errors.uncertainty if origin.depth_errors is not None else None,
        ("origin_loc", "method_id"): origin.method_id.id if origin.method_id is not None else None,
        ("origin_loc", "earth_model_id"): origin.earth_model_id.id if origin.earth_model_id is not None else None}
        
    unc_loc_info = {   
        ("origin_uncertainty_loc", "horizontal_uncertainty"): origin.origin_uncertainty.horizontal_uncertainty if origin.origin_uncertainty is not None else None,
        ("origin_uncertainty_loc", "min_horizontal_uncertainty"): origin.origin_uncertainty.min_horizontal_uncertainty if origin.origin_uncertainty is not None else None,
        ("origin_uncertainty_loc", "max_horizontal_uncertainty"): origin.origin_uncertainty.max_horizontal_uncertainty if origin.origin_uncertainty is not None else None,
        ("origin_uncertainty_loc", "azimuth_max_horizontal_uncertainty"): origin.origin_uncertainty.azimuth_max_horizontal_uncertainty if origin.origin_uncertainty is not None else None,
        ("origin_uncertainty_loc", "semi_major_axis_length"):None,
        ("origin_uncertainty_loc", "semi_minor_axis_length"):None,
        ("origin_uncertainty_loc", "semi_intermediate_axis_length"):None,
        ("origin_uncertainty_loc", "major_axis_plunge"):None,
        ("origin_uncertainty_loc", "major_axis_azimuth"):None,
        ("origin_uncertainty_loc", "major_axis_rotation"):None,
    }
    
    
    if origin.origin_uncertainty is not None:
        if origin.origin_uncertainty.confidence_ellipsoid is not None:
            unc_loc_info[("origin_uncertainty_loc", "semi_major_axis_length")] = origin.origin_uncertainty.confidence_ellipsoid.semi_major_axis_length
            unc_loc_info[("origin_uncertainty_loc", "semi_minor_axis_length")] = origin.origin_uncertainty.confidence_ellipsoid.semi_minor_axis_length
            unc_loc_info[("origin_uncertainty_loc", "semi_intermediate_axis_length")] = origin.origin_uncertainty.confidence_ellipsoid.semi_intermediate_axis_length
            unc_loc_info[("origin_uncertainty_loc", "major_axis_plunge")] = origin.origin_uncertainty.confidence_ellipsoid.major_axis_plunge
            unc_loc_info[("origin_uncertainty_loc", "major_axis_azimuth")] = origin.origin_uncertainty.confidence_ellipsoid.major_axis_azimuth
            unc_loc_info[("origin_uncertainty_loc", "major_axis_rotation")] = origin.origin_uncertainty.confidence_ellipsoid.major_axis_rotation
    
    
    # Combine all information into a single dictionary
    info = ev_info.copy()
    info.update(loc_info)
    info.update(unc_loc_info)
    
    # for x, y in info.items():
    #     print(x, y)
    
    return info
    
def get_custom_info(event):
    """
    Extracts custom information from a seismic event, including origin, picks, 
    and magnitude information.

    Parameters:
    event : Event object
        The seismic event from which to extract information.

    Returns:
    tuple
        A tuple containing:
        - DataFrame with combined origin, picks, and magnitude information.
        - Picks contributions as a dictionary.
        - Magnitude contributions as a dictionary.
    """
    
    # Retrieve custom origin information from the event
    origin_info = get_custom_origin(event)
    
    
    # Retrieve picks information and contributions from the event
    picks_info, picks_contributions = get_custom_arrivals(event)
    
    # Retrieve magnitude information and contributions from the event
    mag_info, mag_contributions = get_custom_pref_mag(event)
    
    # Prepare picks information with a multilevel column structure
    picks_info = {("picks", x): y for x, y in picks_info.items()}
    
    # Prepare magnitude information with a multilevel column structure
    mag_info = {("mag", x): y for x, y in mag_info.items()}
    
    # Update the origin information with picks information
    origin_info.update(picks_info)
    
    # Update the origin information with magnitude information
    origin_info.update(mag_info)
    
    # for x, y in origin_info.items():
    #     print(x, y)
    
    # Convert the combined origin information into a Pandas DataFrame
    origin_info = pd.DataFrame([origin_info])
    # Create a MultiIndex for the columns using the dictionary keys (tuples)
    origin_info.columns = pd.MultiIndex.from_tuples(origin_info.keys())
    
    return origin_info, picks_contributions, mag_contributions

def get_event_ids(catalog):
    """
    Extracts the event IDs from a seismic catalog.

    Parameters:
    catalog : Catalog object
        The catalog containing seismic events.

    Returns:
    list
        A list of event IDs extracted from the catalog.
    """
    
    # Initialize an empty list to hold event IDs
    ev_ids = []
    
    # Iterate through each event in the catalog
    for event in catalog:
        # Retrieve the preferred origin of the event
        pref_origin = event.preferred_origin()
        
        # Extract the event ID from the preferred origin
        eventid = pref_origin.extra.dataid.value
        
        # Append the event ID to the list
        ev_ids.append(eventid)
    
    return ev_ids
        
def save_info(path, info):
    """
    Saves the seismic event information to CSV and SQLite database files.

    Parameters:
    path : str
        The folder path where the information will be saved.
    info : dict
        A dictionary containing seismic event information with keys like 
        'origin', 'picks', 'mags', etc. Each key has an associated DataFrame.
    """
    
    # Iterate through the info dictionary, handling each key-value pair
    for key, value in info.items():
        
        # If the key is 'origin', save it as a CSV file
        if key == "origin":
            info_path = os.path.join(path, f"{key}.csv")
            
            # Save the DataFrame to a CSV file, appending if the file already exists
            value.to_csv(
                info_path, 
                mode='a',  # Append mode
                header=not pd.io.common.file_exists(info_path),  # Add header only if the file doesn't exist
                index=False  # Do not write row numbers
            )
        else:
            # For other keys, save the data in a SQLite database
            info_path = os.path.join(path, f"{key}.db")
            
            # Group the DataFrame by 'ev_id' and iterate over each group
            for ev_id, df_by_evid in value.groupby("ev_id").__iter__():
                
                if not isinstance(ev_id,str):
                    ev_id = str(ev_id)
                    
                
                with sqlite3.connect(info_path) as conn:
                    # Save each group to a SQLite table, appending to the table if it exists
                    df_by_evid.to_sql(
                        ev_id, 
                        conn, 
                        if_exists='append',  # Append data to the table if it exists
                        index=False  # Do not write row numbers
                    )
                        
                # testing...        
                # try:
                
                #     with sqlite3.connect(info_path) as conn:
                #         # if "time" in list(df_by_evid.columns):
                #         #     df_by_evid['time'] = df_by_evid['time'].astype(str)
                #         # df_by_evid.fillna(value='NULL', inplace=True)
                #         # print(key,df_by_evid.info())
                #         # for i,row in df_by_evid.iterrows():
                #         #     print(i,row)
                #         # Save DataFrame to SQLite database, appending if the table exists
                #         # df_by_evid.to_sql(ev_id, conn, if_exists='append', index=False)
                #         df_by_evid.to_sql(ev_id, conn, if_exists='append', index=False)
                # except:
                #     print(ev_id)
                #     with sqlite3.connect(os.path.join(path,f"{key}.bad")) as conn:
                #         df_by_evid.to_csv(os.path.join(path,f"{key}.csv"))
                #         df_by_evid.to_sql(ev_id, conn, if_exists='append', index=False)
                #     # exit()
                        
class CustomClient(FDSNClient):
    """
    A custom client class that extends the base Client class to 
    retrieve seismic event data with additional processing.

    Inherits from:
    FDSN Client
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initializes the CustomClient class by calling the constructor 
        of the base FDSN Client class.

        Parameters:
        *args : variable length argument list
            Positional arguments passed to the base class constructor.
        **kwargs : variable length keyword arguments
            Keyword arguments passed to the base class constructor.
        """
        super().__init__(*args, **kwargs)
        
    def get_custom_events(self, *args, max_events_in_ram=1e6, output_folder=None, **kwargs):
        """
        Retrieves custom seismic event data including origins, picks, 
        and magnitudes.

        Parameters:
        *args : variable length argument list
            Positional arguments passed to the get_events method.
        max_events_in_ram : int, optional, default=1e6
            Maximum number of events to hold in memory (RAM) before stopping or 
            prompting to save the data to disk.
        output_folder : str, optional, default=None
            Folder path where the event data will be saved if provided. If not 
            specified, data will only be stored in memory.
        **kwargs : variable length keyword arguments
            Keyword arguments passed to the get_events method.

        Returns:
        tuple
            A tuple containing:
            - DataFrame of origins for all events.
            - DataFrame of picks for all events.
            - DataFrame of magnitudes for all events.
        """
        # Retrieve the catalog of events using the get_events method
        catalog = self.get_events(*args, **kwargs)

        # Extract event IDs from the catalog
        ev_ids = get_event_ids(catalog)

        # Initialize lists to store origins, picks, and magnitudes
        all_origins, all_picks, all_mags = [], [], []

        # Loop through each event ID to gather detailed event information
        for ev_id in ev_ids[::-1]:
            # Catalog with arrivals. This is a workaround to retrieve 
            # arrivals by specifying the event ID.
            cat = self.get_events(eventid=ev_id)

            # Get the first event from the catalog
            event = cat[0]

            # Extract custom information for the event
            origin, picks, mags = get_custom_info(event)

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
   
if __name__=="__main__":
    from obspy import UTCDateTime


    provider = "USGS"
    client =  CustomClient(provider)
    region = [-104.84329,-103.79942,31.39610,31.91505]
    cat = client.get_custom_events(starttime=UTCDateTime("2024-04-18T23:00:00"),
                            endtime=UTCDateTime("2024-04-19T23:00:00"),
                            minlatitude=region[2], maxlatitude=region[3], 
                            minlongitude=region[0], maxlongitude=region[1],
                            includeallorigins=True,
                            #eventid="tx2024hstr",
                            #includeallmagnitudes=True,
                            )
    print(cat)
    