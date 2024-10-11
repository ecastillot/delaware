# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-09-24 21:02:12
#  * @modify date 2024-09-24 21:02:12
#  * @desc [description]
#  */
import pandas as pd
from datetime import datetime
from delaware.core.eqviewer import Catalog,Stations
import pandas as pd

def get_db_stations(stations_path,lon_lims, lat_lims,proj):
    """Retrieve and filter station data within a specified polygon.

    Args:
        stations_path (str): Path to the CSV file containing station data.
            columns(network,station,latitude,longitude,elevation,)

    Returns:
        Stations: An instance of the Stations class containing filtered stations.
    """
    # Prepare stations data by reading from the CSV file
    stations_data = pd.read_csv(stations_path)
    
    
    # Add an index column for station identification
    stations_data["station_index"] = stations_data.index
    
    # Convert elevation from meters to kilometers
    stations_data["elevation"] = stations_data["elevation"] / 1e3
    
    # Define the polygon coordinates for the region of interest
    x,y = lon_lims,lat_lims
    dw_w_pol = [
        (x[1], y[0]),
        (x[1], y[1]),
        (x[0], y[1]),
        (x[0], y[0]),
        (x[1], y[0])  # Closing the polygon
    ]
    
    # Create an instance of the Stations class with the prepared data
    stations = Stations(stations_data, xy_epsg=proj)
    
    # Filter the stations to include only those within the defined polygon
    stations.filter_region(polygon=dw_w_pol)
    
    return stations

def get_texnet_high_resolution_catalog(path):
    df = pd.read_csv(path)
    # Function to create datetime
    def create_datetime(row):
        try:
            return datetime(year=row['yr'], month=row['mon'], day=row['day'],
                            hour=row['hr'], minute=row['min_'], second=int(row['sec']),
                            microsecond=int((row['sec'] % 1) * 1e6))
        except ValueError:
            return pd.NaT  # or use `None` if you want to see errors

    # Apply function to create 'origin_time'
    df['origin_time'] = df.apply(create_datetime, axis=1)
    df = df.rename(columns={"latR":"latitude",
                            "lonR":"longitude",
                            "depR":"depth",
                            "mag":"magnitude",
                            "EventId":"id"})
    
    catalog = Catalog(df)
    return catalog
    
    # df.to_csv(outpath,index=False)
    
