import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth

def filter_data(data,key,start=None,end=None,reset_index=True):
    """
    Filter data of the catalog.

    Parameters:
    -----------
    key: str
        Name of the column to filter
    start: int or float or datetime.datetime
        must be the same type as data[key] does
    end: int or float or datetime.datetime
        must be the same type as data[key] does
    
    """
    if (start is not None) and (len(data) !=0):
        data = data[data[key]>=start]
    if (end is not None) and (len(data) !=0):
        data = data[data[key]<=end]
    
    if reset_index:
        data.reset_index(drop=True,inplace=True)
    return data

def filter_data_by_r_az(data, latitude, longitude, r, az=None,reset_index=True):
        """
        Filter data points based on distance (r) and optionally azimuth (az).

        Parameters:
        ----------
        latitude : float
            Latitude of the reference point.
        longitude : float
            Longitude of the reference point.
        r : float
            Maximum distance in kilometers to filter data points.
        az : float, optional
            Maximum azimuth in degrees to filter data points (default is None).
        
        Returns:
        -------
        self : object
            The object with updated data after filtering.
        """
        if data.empty:
            return data
        
        # Calculate distance, azimuth, and back-azimuth from the reference point
        # to each data point (latitude, longitude).
        is_in_polygon = lambda x: gps2dist_azimuth(
                                latitude, longitude, x.latitude, x.longitude
                                )
        data = data.copy()
        
        # Apply the 'is_in_polygon' function to each row in the DataFrame.
        # This results in a Series of tuples (r, az, baz) for each data point.
        mask = data[["longitude", "latitude"]].apply(is_in_polygon, axis=1)
        
        # Convert the Series of tuples into a DataFrame with columns 'r' (distance), 
        # 'az' (azimuth), and 'baz' (back-azimuth).
        mask = pd.DataFrame(mask.tolist(), columns=["r", "az", "baz"])
        
        # Convert distance 'r' from meters to kilometers.
        mask.loc[:, "r"] /= 1e3
        
        
        data[mask.columns.to_list()] = mask
        
        data = data[data["r"] < r]
        
        if az is not None:
            data = data[data["az"] < az]
        
        if reset_index:
            data.reset_index(drop=True,inplace=True)
        
        # Return the updated object to allow method chaining.
        return data