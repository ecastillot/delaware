
from obspy.clients.fdsn import Client 
import pandas as pd

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
            "network_code": pick.waveform_id.network_code,
            "station_code": pick.waveform_id.station_code,
            "location_code": pick.waveform_id.location_code,
            "channel_code": pick.waveform_id.channel_code,
            "phase_hint": pick.phase_hint,
            "time": pick.time.datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "time_lower_error": pick.time_errors.lower_uncertainty,
            "time_upper_error": pick.time_errors.upper_uncertainty,
            "author": pick.creation_info.author,
            "filter_id": pick.filter_id,
            "method_id": pick.method_id,
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
            "network_code": sta_mag.waveform_id.network_code,
            "station_code": sta_mag.waveform_id.station_code,
            "location_code": sta_mag.waveform_id.location_code,
            "channel_code": sta_mag.waveform_id.channel_code,
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
    origin = event.preferred_origin()
    
    # Get origin quality information
    info = dict(origin.quality)
    
    # Retrieve custom picks
    picks = get_custom_picks(event)
    
    arr_contributions = {}
    
    # Loop through each arrival in the origin
    for arrival in origin.arrivals:
        pick_info = picks[arrival.pick_id.id]
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
    magnitude = event.preferred_magnitude()
    
    # Get preferred magnitude information
    info = {
        "mag": magnitude.mag,
        "uncertainty": magnitude.mag_errors.uncertainty,
        "type": magnitude.magnitude_type,
        "method_id": magnitude.method_id.id.split("/")[-1],
        "station_count": magnitude.station_count,
        "evaluation_status": magnitude.evaluation_status,
    }
    
    # Retrieve station magnitudes
    sta_mags = get_custom_station_magnitudes(event)
    
    mag_contributions = {}
    
    # Loop through each station magnitude contribution
    for used_sta_mag in magnitude.station_magnitude_contributions:
        sta_info = sta_mags[used_sta_mag.station_magnitude_id.id]
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
        ("event", "id"): event.resource_id.id.split("/")[-1],
        ("event", "type"): event.event_type,
        ("event", "type_certainty"): event.event_type_certainty,
        ("event", "agency"): origin.creation_info.agency_id,
        ("event", "evaluation_mode"): origin.evaluation_mode,
        ("event", "evaluation_status"): origin.evaluation_status
    }
    
    # Prepare location information
    loc_info = {
        ("loc", "origin_time"): origin.time.datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ("loc", "longitude"): origin.longitude,
        ("loc", "longitude_error"): origin.longitude_errors.uncertainty,
        ("loc", "latitude"): origin.latitude,
        ("loc", "latitude_error"): origin.latitude_errors.uncertainty,
        ("loc", "depth"): origin.depth,
        ("loc", "depth_error"): origin.depth_errors.uncertainty,
        ("loc", "method_id"): origin.method_id.id,
        ("loc", "earth_model_id"): origin.earth_model_id.id,
    }
    
    # Prepare method information from origin uncertainty
    method_info = {("method", x): y for x, y in dict(origin.origin_uncertainty).items()}
    
    # Prepare creation information
    creation_info = {("creation", x): y for x, y in dict(origin.creation_info).items()}
    
    # Combine all information into a single dictionary
    info = ev_info.copy()
    info.update(loc_info)
    info.update(method_info)
    info.update(creation_info)
    
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
    
    # Convert the combined origin information into a Pandas DataFrame
    origin_info = pd.DataFrame.from_dict(origin_info)
    
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
        
    def get_custom_events(self, *args, **kwargs):
        """
        Retrieves custom seismic event data including origins, picks, 
        and magnitudes.

        Parameters:
        *args : variable length argument list
            Positional arguments passed to the get_events method.
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
            
            # Append the retrieved information to respective lists
            all_origins.append(origin)
            all_picks.append(picks)
            all_mags.append(mags)
        
        # Concatenate the data from all events if multiple events are found
        if len(ev_ids) > 1:
            all_origins = pd.concat(all_origins, axis=0)
            all_picks = pd.concat(all_picks, axis=0)
            all_mags = pd.concat(all_mags, axis=0)
        else:
            # If only one event is found, keep the single DataFrame
            all_origins = all_origins[0]
            all_picks = all_picks[0]
            all_mags = all_mags[0]
            
        return all_origins, all_picks, all_mags
        
        