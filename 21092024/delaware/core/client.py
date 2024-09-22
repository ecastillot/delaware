
from obspy.clients.fdsn import Client 
import pandas as pd
import os
import sqlite3

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
            "network_code": pick.waveform_id.network_code if pick.waveform_id is not None else None,
            "station_code": pick.waveform_id.station_code if pick.waveform_id is not None else None,
            "location_code": pick.waveform_id.location_code if pick.waveform_id is not None else None,
            "channel_code": pick.waveform_id.channel_code if pick.waveform_id is not None else None,
            "phase_hint": pick.phase_hint,
            "time": pick.time.datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
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
        "mag": magnitude.mag,
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
        ("event", "id"): event.resource_id.id.split("/")[-1] if event.resource_id is not None else None,
        ("event", "type"): event.event_type,
        ("event", "type_certainty"): event.event_type_certainty,
        ("event", "agency"): origin.creation_info.agency_id,
        ("event", "evaluation_mode"): origin.evaluation_mode,
        ("event", "evaluation_status"): origin.evaluation_status
    }
    
    # Prepare location information
    loc_info = {
        ("loc", "origin_time"): origin.time.datetime.strftime("%Y-%m-%d %H:%M:%S.%f") if origin.time is not None else None,
        ("loc", "longitude"): origin.longitude,
        ("loc", "longitude_error"): origin.longitude_errors.uncertainty,
        ("loc", "latitude"): origin.latitude,
        ("loc", "latitude_error"): origin.latitude_errors.uncertainty,
        ("loc", "depth"): origin.depth,
        ("loc", "depth_error"): origin.depth_errors.uncertainty if origin.depth_errors is not None else None,
        ("loc", "method_id"): origin.method_id.id if origin.method_id is not None else None,
        ("loc", "earth_model_id"): origin.earth_model_id.id if origin.earth_model_id is not None else None,
    }
    
    
    # Prepare method information from origin uncertainty
    method_info = {("method", x): y for x, y in dict(origin.origin_uncertainty).items() if x != "confidence_ellipsoid"}
    method_info2 = {("method", x): y for x, y in dict(origin.origin_uncertainty.confidence_ellipsoid).items()}
    method_info.update(method_info2)
    
    # Prepare creation information
    creation_info = {("creation", x): y for x, y in dict(origin.creation_info).items()}
    
    # Combine all information into a single dictionary
    info = ev_info.copy()
    info.update(loc_info)
    info.update(method_info)
    info.update(creation_info)
    
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
        

def save_info(path,info):
    
    for key, value in info.items():
        
        if key == "origin":
            info_path = os.path.join(path,f"{key}.csv")
            # Save the dataframes to files, appending if they already exist
            value.to_csv(info_path, mode='a', 
                        header=not pd.io.common.file_exists(info_path), 
                        index=False)
        else:
            info_path = os.path.join(path,f"{key}.db")
            
            for ev_id,df_by_evid in value.groupby("ev_id").__iter__():
                with sqlite3.connect(info_path) as conn:
                    # Save DataFrame to SQLite database, appending if the table exists
                    df_by_evid.to_sql(ev_id, conn, if_exists='append', index=False)
                        
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
                        
class CustomClient(Client):
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
        for ev_id in ev_ids:
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
    