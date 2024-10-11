# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-02-09 17:37:34
#  * @modify date 2024-02-09 17:37:34
#  * @desc [description]
#  */

import numpy as np
import h5py
import copy
import matplotlib
import os
import random
import pandas as pd
import pykonal
import time
import matplotlib.pyplot as plt
import concurrent.futures as cf
from . import tt_utils as ttu
import cartopy.crs as ccrs
import cartopy.feature as cfeature
### TravelTime

class TravelTime(object):
    def __init__(self, data: pd.DataFrame, dropinf:bool=True) -> None:
        """
        Initialize the TravelTime object.

        Args:
            data (pd.DataFrame): DataFrame containing travel time data.
            dropinf (bool,Optional): Remove infinite values. Default True
        """
        # Define mandatory columns
        self._mandatory_columns = ['event_index', 'src_lon', 'src_lat', 'src_z[km]',
                                   "station_index", "travel_time"]
        
        # Check if all mandatory columns are present in the DataFrame
        check = all(item in data.columns.to_list() for item in self._mandatory_columns)
        if not check:
            raise Exception("There is not the mandatory columns for the data in Earthquakes object." \
                            + f"->{self._mandatory_columns}")
        if dropinf:
            # Replace infinite updated data with nan
            data.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Drop rows with NaN
            data.dropna(inplace=True)
            
        data = data.sort_values(by="travel_time",ignore_index=True)
        self.data = data
    
    @property
    def n_earthquakes(self):
        """
        Get the number of unique earthquakes in the travel time data.

        Returns:
            int: Number of unique earthquakes.
        """
        _data = self.data["event_index"].copy()
        _data.drop_duplicates(inplace=True)
        return len(_data)
    
    @property
    def n_stations(self):
        """
        Get the number of unique stations in the travel time data.

        Returns:
            int: Number of unique stations.
        """
        _data = self.data["station_index"].copy()
        _data.drop_duplicates(inplace=True)
        return len(_data)
    
    @property
    def n_traveltimes(self):
        """
        Get the total number of travel times in the data.

        Returns:
            int: Total number of travel times.
        """
        return len(self.data)
    
    def __str__(self) -> str:
        """
        Get a string representation of the TravelTime object.

        Returns:
            str: String representation.
        """
        msg = f"TravelTime ({self.n_traveltimes})\n\tEarthquakes({self.n_earthquakes}) - Stations({self.n_stations})"
        return msg
    
    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)
    
    def merge_stations(self, stations: ttu.Stations, inplace: bool = False):
        """
        Merge the stations of the current object with another Stations object.

        Args:
            stations (ttu.Stations): The Stations object containing stations to merge.
            inplace (bool): Whether to modify the current object in-place or return a new merged object. Default is True.

        Returns:
            pd.DataFrame or None: Merged DataFrame if inplace is False, else None.
        """
        # # Extract data from the provided Stations object
        # stations_data = stations.data
        
        # Merge the stations using station_index as the key
        data = pd.merge(self.data, stations.data, how="inner", on="station_index")
        
        # Update the data attribute based on the inplace parameter
        if inplace:
            self.data = data
        else:
            return data
        
    def plot(self, stations: ttu.Stations, event_index: int = None, 
                show: bool = True, savefig: str = None):
        """
        Plot travel times for a given earthquake event.

        Args:
            stations (ttu.Stations): Stations object containing station data.
            event_index (int, optional): ID of the earthquake event. If not provided, a random event ID will be selected. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to True.
            savefig (str, optional): File path to save the plot. Defaults to None.

        Returns:
            matplotlib.figure.Figure: The generated plot.
        """
        # First, obtain travel time data using 'get_traveltime' function
        tt = self.data
        stations_data = stations.data

        if event_index is None:
            event_index = random.randint(min(tt["event_index"]), max(tt["event_index"]))
        print(event_index, len(tt))

        # Filter travel time data for the given earthquake event ID
        df = tt[tt["event_index"] == event_index]
        df = df.reset_index()
        
        # print(df)
        # exit()
        #  getting the source
        source = df.loc[0,["src_lon","src_lat","src_z[km]"]]
        # print(df)
        # exit()
        
        if "station" in df.columns.to_list():
            pass
        else:
            df = pd.merge(df, stations_data, how="inner", on="station_index")

        # Create the plot
        fig = plt.figure(figsize=(12, 10))
        # ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # # Add features
        # ax1.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='red')
        # ax1.add_feature(cfeature.COASTLINE)

        ax1 = fig.add_subplot(1, 1, 1)
        # Scatter plots
        ax1.scatter(source.src_lon, source.src_lat, color='red', label='Source', s=100, edgecolor='black', zorder=5)
        sc = ax1.scatter(df['longitude'], df['latitude'], 
                        c=df['travel_time'], cmap='viridis', 
                        label='Station Locations', 
                        marker='^', s=100, edgecolor='black', zorder=4)

        # Calculate axis ranges
        x_range = df['longitude'].max() - df['longitude'].min()
        y_range = df['latitude'].max() - df['latitude'].min()

        # Determine colorbar orientation based on axis ranges
        if x_range > y_range:
            cbar = plt.colorbar(sc, ax=ax1, 
                                orientation='horizontal', pad=0.1, aspect=50)
        else:
            cbar = plt.colorbar(sc, ax=ax1, 
                                orientation='vertical', pad=0.05)
            
        cbar.set_label('Travel Time', fontsize=14)
        cbar.ax.yaxis.set_tick_params(color='black', size=8, width=2)  # Customize colorbar ticks
        cbar.ax.tick_params(labelsize=12)  # Set colorbar tick labels to be larger

        # Set labels and title
        ax1.set_xlabel('Longitude [°]', fontsize=14)
        ax1.set_ylabel('Latitude [°]', fontsize=14)
        ax1.set_title(f'Event {event_index} | Lon:{round(source.src_lon,4)} Lat:{round(source.src_lat,4)} Z: {round(source["src_z[km]"],4)}')

        # Customize grid
        ax1.grid(True, linestyle='--', color='gray', linewidth=0.5)
        
        # Adjust font size for axis tick labels
        ax1.tick_params(axis='both', labelsize=12)  # Set x and y axis tick labels to be larger

        # Set aspect ratio to be equal
        ax1.set_aspect('equal', 'box')

        # Add legend
        # ax1.legend(loc='upper right', fontsize=12)
        # Add legend outside the figure, top left
        ax1.legend(loc='lower right', bbox_to_anchor=(1, -0.22), 
                   fontsize=12, ncol=2)
        
        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust right margin to fit the legend

        # Save the plot if savefig is specified
        if savefig is not None:
            fig.savefig(savefig)

        # Show the plot if specified
        if show:
            plt.show()

        return fig
    
## Earthquake TravelTime 

def get_station_location_on_grid(x: float, y: float, z: float, field: ttu.VelModel):
    """
    Get the grid location of a station in the velocity model field.

    Args:
        x (float): X-coordinate of the station.
        y (float): Y-coordinate of the station.
        z (float): Z-coordinate of the station.
        field (ttu.VelModel): Velocity model field.

    Returns:
        tuple: Grid location of the station (x_index, y_index, z_index).
    """
    # Get node intervals
    dx, dy, dz = field.node_intervals
    
    # Get minimum coordinates
    x0, y0, z0 = field.min_coords
    
    # Calculate grid indices
    x_index = round((x - x0) / dx)
    y_index = round((y - y0) / dy)
    z_index = round((z - z0) / dz)
    
    return x_index, y_index, z_index

def get_tt_from_single_source(source: ttu.Source,
                              stations: ttu.Stations,
                              vel_model: ttu.VelModel):
    """
    Calculate travel times from a single seismic source to multiple stations.

    Args:
        source (ttu.Source): Seismic source.
        stations (ttu.Stations): Seismic stations.
        vel_model (ttu.VelModel): Velocity model.

    Returns:
        pd.DataFrame: DataFrame containing travel times.
    """
    # Initialize PointSourceSolver
    solver = pykonal.solver.PointSourceSolver(coord_sys=vel_model.coord_sys)
    
    # Set velocity model attributes
    solver.vv.min_coords = vel_model.min_coords
    solver.vv.node_intervals = vel_model.node_intervals
    solver.vv.npts = vel_model.npts
    solver.vv.values = vel_model.values
    
    # Set source location
    solver.src_loc = (source.x, source.y, source.z)
    
    # Solve for travel times
    solver.solve()
    tt = solver.tt.values
    
    # Initialize receivers dictionary
    receivers = {"station_index": [],
                 "travel_time": []}
    
    # Iterate over stations
    for i, row in stations.data.iterrows():
        # print(i,row)
        # Get station's grid location
        sta_grid_loc = get_station_location_on_grid(row["x[km]"],
                                                    row["y[km]"],
                                                    row["z[km]"],
                                                    vel_model)
        
        # Append station index and travel time to receivers dictionary
        receivers["station_index"].append(row.station_index)
        receivers["travel_time"].append(tt[sta_grid_loc])
    
    # Create DataFrame from receivers dictionary
    df = pd.DataFrame.from_dict(receivers)
    
    # Add source coordinates to DataFrame
    df["src_lon"] = source.longitude
    df["src_lat"] = source.latitude
    df["src_x[km]"] = source.x
    df["src_y[km]"] = source.y
    df["src_z[km]"] = source.z

    # Reorder columns in DataFrame
    df = df[['src_lon', 'src_lat', 'src_x[km]', 'src_y[km]', 'src_z[km]',
             'station_index', 'travel_time']]
    
    return df

def read_traveltime_from_earthquake(input: str, event_id: int, dropinf: bool = True):
    """
    read travel time from an HDF5 earthquake file.

    Args:
        input (str): Input file path to the HDF5 file.
        event_id (int): Event ID for which to read travel times.
        dropinf (bool,Optional): Remove infinite values. Default True

    Returns:
        TravelTime: Travel time data for the specified event.
    """
    # Open the HDF5 file in read mode
    hf = h5py.File(input, 'r')
    
    # Get earthquake data for the current event
    eq = hf.get(f"earthquake_{event_id}")
    eq_dict = {}
    
    # Extract data from HDF5 dataset to a dictionary
    for key in eq.keys():
        
        # Ensuring it gets only earthquake info
        if "earthquake" not in key:
            continue
        
        eq_dict[key] = eq[key][:]
    
    # Convert dictionary to DataFrame
    df = pd.DataFrame(eq_dict)
    
        
    # Create TravelTime object from earthquake data
    tt = TravelTime(df,dropinf)
    
    return tt

class EarthquakeTravelTime(object):
    def __init__(self, phase: str, stations: ttu.Stations, earthquakes: ttu.Earthquakes):
        """
        Initialize the PhaseTravelTime object.

        Args:
            phase (str): Phase type.
            stations (ttu.Stations): Seismic stations.
            earthquakes (ttu.Earthquakes): Earthquake data.
        """
        self.phase = phase
        self.stations = stations
        self.earthquakes = earthquakes
    
    def load_velocity_model(self, path: str, xy_epsg: str):
        """
        Load a velocity model from an NPZ file.

        Args:
            path (str): The path to the NPZ file containing the velocity model.
            xy_epsg (str): The EPSG code specifying the coordinate reference system.

        Returns:
            None
        """
        self.model = ttu.VelModel.load_npz(path, self.phase, xy_epsg)
        return self.model
        
    def add_grid_with_velocity_model(self, x: tuple, y: tuple, z: tuple,
                                     nx: int, ny: int, nz: int,
                                     xy_epsg: str,
                                     vel1d: dict = {"depth":[-10,4,25,32,40,100,200],
                                                     "vel":[4.8,6.6,7,8,8.1,8.2,8.2]},
                                     layer: bool = True):
        """
        Add a grid with a velocity model.

        Args:
            x (tuple): X-coordinates.
            y (tuple): Y-coordinates.
            z (tuple): Z-coordinates.
            nx (int): Number of grid points in the x-direction.
            ny (int): Number of grid points in the y-direction.
            nz (int): Number of grid points in the z-direction.
            xy_epsg (str): EPSG code specifying the coordinate reference system.
            vel1d (dict): Dictionary containing 1D velocity model parameters. Default is a standard model.
            layer (bool): Whether to use layered velocity model. Default is True.

        Returns:
            ttu.VelModel: Velocity model.
        """
        self.model = ttu.get_xyz_velocity_model(x, y, z, nx, ny, nz, xy_epsg,
                                                 self.phase, vel1d, layer)
        return self.model
        
    def get_traveltimes(self, merge_stations: bool = False, output: str = None):
        """
        Get travel times for all earthquakes. First, you have to define your grid with 'add_grid_with_velocity_model' function.

        Args:
            merge_stations (bool, optional): Whether to merge stations with the current object's stations. Default is False.
            output (str): Output file path to store travel time data. Default is None.

        Returns:
            pd.DataFrame: DataFrame containing travel times.
        """
        # Create directory if output path is provided
        if output is not None:
            if not os.path.isdir(os.path.dirname(output)):
                os.makedirs(os.path.dirname(output))
                
        # Initialize HDF5 file for output
        if output is not None:
            hf = h5py.File(output, 'w')
        
        # Initialize empty list for storing travel time DataFrames
        tt_list = []
        
        # Iterate over earthquakes
        xy_epsg = self.earthquakes.xy_epsg
        for i, row in self.earthquakes.data.iterrows():
            # Create Source object
            source = ttu.Source(row.latitude, row.longitude, row.depth,
                                xy_epsg=xy_epsg, origin_time=row.origin_time)
            
            # Calculate travel times for current earthquake
            df = get_tt_from_single_source(source, self.stations, self.model)
            
            # print("Aca",df)
            # Add earthquake ID to DataFrame
            df["event_index"] = i
            
            # integer columns
            df['station_index'] = df['station_index'].astype(int)
            df['event_index'] = df['event_index'].astype(int)
            
            # Reorder columns in DataFrame
            df = df[df.columns[-1:].tolist() + df.columns[:-1].tolist()]
            
            # Append DataFrame to travel time list
            tt_list.append(df)
            
            # Write data to HDF5 file
            if output is not None:
                data_dict = df.to_dict()
                g = hf.create_group(f'earthquake_{str(i)}')
                for key, col in data_dict.items():
                    values = list(col.values())
                    g.create_dataset(key, data=values)
        
        # Concatenate travel time DataFrames
        tt_df = pd.concat(tt_list)
        tt = TravelTime(tt_df)
        
        # # print(tt)
        # for i,gp in tt.data.sort_values("event_index").groupby("event_index").__iter__():
        #     print(i,gp.sort_values("station_index"))
        #     print(self.stations.data)
        
        # Merge stations if specified
        if merge_stations:
            tt.merge_stations(self.stations, inplace=True) 
        
        # Close HDF5 file if opened
        if output is not None:
            hf.close()
            eq = tt_df[["event_index","src_lat","src_lon","src_z[km]"]] 
            eq = eq.drop_duplicates("event_index",ignore_index=True)
            eq_path = ttu.change_file_extension(output,".csv")
            eq.to_csv(eq_path,index=False)
        
        return tt

    def read_traveltimes_from_single_earthquake(self, input: str, event_id: int, 
                                                merge_stations: bool = False, dropinf: bool = True):
        """
        read travel time data from an HDF5 dataset for a single earthquake event.

        Args:
            input (str): Input file path to the HDF5 file containing earthquake data.
            event_id (int): Event ID of the earthquake to read travel time data for.
            merge_stations (bool, optional): Whether to merge stations with the current object's stations. Default is False.
            dropinf (bool, optional): Whether to remove infinite values. Default is True.

        Returns:
            TravelTime: Travel time data for the specified earthquake event.
        """
        # Call the read_traveltime_from_earthquake function to read travel time data
        tt = read_traveltime_from_earthquake(input, event_id, dropinf)
        
        # Merge stations if specified
        if merge_stations:
            tt.merge_stations(self.stations, inplace=True) 
        
        return tt
    
## TravelTime Dataset from grid points (massive travel times)   
 
class Station2Source():
    def __init__(self, vel_model: ttu.VelModel,
                 tt_data: pd.DataFrame):
        """
        Initialize Station2Source object.

        Args:
            vel_model (ttu.VelModel): Velocity model object.
            tt_data (pd.DataFrame): DataFrame containing travel time data.
        """
        self.tt = tt_data
        self.vel_model = vel_model

    def tt2df(self):
        """
        Convert travel time data to DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing travel time data.
        """
        
        # Create 2D array for travel time data.
        all = []
        # for x in range(self.tt.shape[0]):
        #     for y in range(self.tt.shape[1]):
        #         for z in range(self.tt.shape[2]):
        #             all.append((x,y,z,self.tt[y][x][z]))
        # print(self.tt.shape)
        # exit()
        for x in range(self.tt.shape[0]):
            for y in range(self.tt.shape[1]):
                for z in range(self.tt.shape[2]):
                    # print(x,y,z)
                    all.append((x,y,z,self.tt[x][y][z]))
        
        # THis approach could be better, but I have not had time to test it,
        # so for now I will use the for loop
        
        # indices = np.array(list(np.ndindex(self.tt.shape)))
        # values = self.tt[indices[:, 1], indices[:, 0], indices[:, 2]]
        # all = np.column_stack((indices, values))
        
        # print(self.tt.shape)
        columns = ['src_lon', 'src_lat', 'src_z[km]', 'travel_time']
        
        #  # Create DataFrame from array
        df = pd.DataFrame(all, columns=columns)
        
        # print(df)
        
        # Extract geographical coordinates from velocity model
        lon, lat, z = self.vel_model.geo_coords
        # print(lon,lat,z)
        
        # Convert source coordinates to geographical coordinates
        df["src_lon"] = df["src_lon"].apply(lambda x: lon[int(x)])
        df["src_lat"] = df["src_lat"].apply(lambda x: lat[int(x)])
        df["src_z[km]"] = df["src_z[km]"].apply(lambda x: z[int(x)])
        
        # # Create a 3D plot to test
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        
        # # for zz,dfz in df.groupby("src_z[km]"):
        # #     print(zz)

        #     # Scatter plot
        # sc = ax.scatter(df['src_lon'], df['src_lat'],df['src_z[km]']*-1,
        #                 c=df['travel_time'], 
        #                 cmap='viridis')

        # # Add color bar which maps values to colors
        # plt.colorbar(sc, label='Travel Time')

        # # Set labels
        # ax.set_xlabel('Longitude')
        # ax.set_ylabel('Latitude')
        # # ax.set_zlabel('Altitude (km)')
        

        # # Show plot
        # plt.show()
        # exit()
        
        return df

def get_tt_from_grid_points(stations: ttu.Stations,
                            vel_model: ttu.VelModel,
                            polygons: list=[],
                            output: str = None):
    """
    Calculate travel times from grid points to stations.

    Args:
        stations (ttu.Stations): Stations object containing station data.
        vel_model (ttu.VelModel): Velocity model object.
        polygons (list): List of polygons to check if points are inside.
        output (str, optional): Output file path to store travel time data. Default is None.

    Returns:
        pd.DataFrame: DataFrame containing travel time data only for points inside the polygons.
    """
    # Create output directory if specified
    if output is not None:
        if not os.path.isdir(os.path.dirname(output)):
            os.makedirs(os.path.dirname(output))
        hf = h5py.File(output, 'w')
        
        model_path = ttu.change_file_extension(output,".npz")
        vel_model.export_npz(model_path)

    def solver(iterrow,srcs_in_pol):
        station_index, row = iterrow
        print(f"{row.network}.{row.station} ok")

        # Initialize solver
        model = vel_model
        solver = pykonal.solver.PointSourceSolver(coord_sys=model.coord_sys)
        solver.vv.min_coords = model.min_coords
        solver.vv.node_intervals = model.node_intervals
        solver.vv.npts = model.npts
        solver.vv.values = model.values

        # Set source location
        # source = np.array([row["x[km]"], row["y[km]"], row["z[km]"]])
        source = np.array([row["x[km]"], row["y[km]"], row["z[km]"]])
        solver.src_loc = source
        solver.solve()
        
        # print(model.npts)
        # print(model.values)
        # -11593311.0,3750558.3
        # Get travel time values
        values = solver.tt.values
        # print(source,model.node_intervals,model.npts)
        # print(model.min_coords)
        # print(model.max_coords)
        # print(values)
        # exit()
        # fig = plt.figure(figsize=(6, 2.5))
        # ax = fig.add_subplot(1, 1, 1, frameon=False)
        # print(source)
        # print(solver.tt.nodes.shape)
        # print(solver.tt.values.shape)
        # qmesh = ax.pcolormesh(
        #         solver.tt.nodes[:,:,0,0], 
        #         solver.tt.nodes[:,:,0,1], 
        #         solver.tt.values[:,:,0],
        #         cmap=plt.get_cmap("hot")
        #     )
        # plt.show()
        # cbar = fig.colorbar(qmesh, ax=(ax1, ax2))
        # cbar.set_label("Velocity [km/s]")
                
        # Convert travel times to DataFrame
        s2s = Station2Source(vel_model=vel_model, tt_data=values)
        df = s2s.tt2df()
        
        # filtering events. Only events inside the polygon
        mask = srcs_in_pol["inside_pol"]
        df = df[mask]
        
        # Reset index and add event index column
        df = df.reset_index(drop=True)
        df["event_index"] = df.index
        df = df[["event_index", 'src_lon', 'src_lat', 'src_z[km]', 'travel_time']]
        
        df["station_index"] = row.station_index

        # integer columns
        df['station_index'] = df['station_index'].astype(int)
        df['event_index'] = df['event_index'].astype(int)

        # Station index will be the first column
        df = df[df.columns[-1:].tolist() + df.columns[:-1].tolist()]

        # Write data to HDF5 file if specified
        if output is not None:
            #saving earthquake info
            if nsta == 0:
                eq = df[["event_index","src_lat","src_lon","src_z[km]"]]
                eq_path = ttu.change_file_extension(output,".csv")
                try:
                    eq.to_csv(eq_path, index=False)
                    del eq
                except Exception as e:
                    print("Error occurred while saving CSV:", e)
                
            try:   
                data_dict = df.to_dict()
                g = hf.create_group(f'station_{str(station_index)}')
                for key, col in data_dict.items():
                    values = list(col.values())
                    g.create_dataset(key, data=values)
            except Exception as e:
                    print("Error occurred while saving H5:", e)
                
            del df
        else:
            return df

    # selecting sources inside the polygon
    srcs_in_pol = vel_model.filter_geo_coords(polygons)

    # Iterate over stations
    iterrows = stations.data.iterrows()
    
    # Simple loop
    if output is not None:
        for nsta,iterrow in enumerate(iterrows):
            # print(iterrow)
            # exit()
            solver(iterrow,srcs_in_pol)
    else:
        tts = []
        for nsta,iterrow in enumerate(iterrows):
            tt = solver(iterrow,srcs_in_pol)
            tts.append(tt)
        tts = pd.concat(tts)
        return tts

    # ## Threads #it's not working to save h5py files
    # with cf.ThreadPoolExecutor() as executor:
    #     if output is not None:
    #         executor.map(solver, iterrows)
            
    #         # Close HDF5 file if opened
    #         hf.close()
            
    #     else:
    #         tts = list(executor.map(solver, iterrows))

    #         # Concatenate travel time DataFrames
    #         tts = pd.concat(tts)
    #         return tts

def read_traveltime_from_dataset(input: str, event_id: int, dropinf: bool = True):
    """
    read travel time from an HDF5 dataset file.

    Args:
        input (str): Input file path to the HDF5 file.
        event_id (int): Event ID for which to read travel times.
        dropinf (bool,Optional): Remove infinite values. Default True

    Returns:
        TravelTime: Travel time data for the specified event.
    """
    # Open the HDF5 file in read mode
    hf = h5py.File(input, 'r')
    
    # Initialize list to store receiver DataFrames
    receivers = []
    
    # Iterate over stations in the HDF5 file
    for station in hf.keys():
        
        # Ensuring it gets only stations info
        if "station" not in station:
            continue
        
        # Get earthquake data for the current station
        eq = hf.get(station)
        station_dict = {}
        
        # Extract data from HDF5 dataset to a dictionary
        for key in eq.keys():
            station_dict[key] = eq[key][:]
        
        # Convert dictionary to DataFrame
        df = pd.DataFrame(station_dict)
        
        # Filter data for the specified event ID and append to receivers list
        receiver = df[df["event_index"] == event_id]
        receivers.append(receiver)
    
    # Concatenate receiver DataFrames
    receivers = pd.concat(receivers)
    
    # Create TravelTime object from receiver data
    tt = TravelTime(receivers,dropinf)
    return tt
    
class EarthquakeTravelTimeDataset(object):
    def __init__(self, phase: str, 
                 stations: ttu.Stations, 
                 polygons:list = []):
        """
        Initialize the EarthquakeDataset object.

        Args:
            phase (str): Phase type.
            stations (ttu.Stations): Stations object containing station data.
            polygons (list): List of polygons to locate the earthquakes.
        """
        self.phase = phase
        self.stations = stations
        self.polygons = polygons
    
    def load_velocity_model(self, path: str, xy_epsg: str):
        """
        Load a velocity model from an NPZ file.

        Args:
            path (str): The path to the NPZ file containing the velocity model.
            xy_epsg (str): The EPSG code specifying the coordinate reference system.

        Returns:
            None
        """
        self.model = ttu.VelModel.load_npz(path, self.phase, xy_epsg)
        return self.model
    
    def add_grid_with_velocity_model(self, x: tuple, y: tuple, z: tuple,
                                     nx: int, ny: int, nz: int,
                                     xy_epsg: str,
                                     vel1d: dict = {"depth":[-10,4,25,32,40,100,200],
                                                     "vel":[4.8,6.6,7,8,8.1,8.2,8.2]},
                                     layer: bool = True):
        """
        Add a grid with a velocity model.

        Args:
            x (tuple): X-coordinates.
            y (tuple): Y-coordinates.
            z (tuple): Z-coordinates.
            nx (int): Number of grid points in the x-direction.
            ny (int): Number of grid points in the y-direction.
            nz (int): Number of grid points in the z-direction.
            xy_epsg (str): EPSG code specifying the coordinate reference system.
            vel1d (dict): Dictionary containing 1D velocity model parameters. Default is a standard model.
            layer (bool): Whether to use layered velocity model. Default is True.

        Returns:
            ttu.VelModel: Velocity model.
        """
        self.model = ttu.get_xyz_velocity_model(x, y, z, nx, ny, nz, xy_epsg,
                                                 self.phase, vel1d, layer)
        return self.model
    
    def save_traveltimes(self, output: str):
        """
        Save travel times for all grid points considered as earthquakes.

        Args:
            output (str): Output file path to store travel time data.

        """
        get_tt_from_grid_points(stations=self.stations,
                                vel_model=self.model, 
                                output = output,
                                polygons=self.polygons)

    def read_traveltimes_from_single_earthquake(self, input: str, event_id: int, 
                                                merge_stations: bool = False, dropinf: bool = True):
        """
        read travel time data from an HDF5 dataset for a single earthquake event.

        Args:
            input (str): Input file path to the HDF5 file containing earthquake data.
            event_id (int): Event ID of the earthquake to read travel time data for.
            merge_stations (bool, optional): Whether to merge stations with the current object's stations. Default is False.
            dropinf (bool, optional): Whether to remove infinite values. Default is True.

        Returns:
            TravelTime: Travel time data for the specified earthquake event.
        """
        # Call the read_traveltime_from_dataset function to read travel time data
        tt = read_traveltime_from_dataset(input, event_id, dropinf)
        
        # Merge stations if specified
        if merge_stations:
            tt.merge_stations(self.stations, inplace=True) 
        
        return tt
    
        
if __name__ == "__main__":
    
    stations_path = pd.read_csv("/home/emmanuel/ecastillo/dev/associator/GPA/data/CM_stations.csv")
    stations = ttu.Stations(stations_path,"EPSG:3116")
    
    earthquakes_data = pd.DataFrame.from_dict({"latitude":[6.81],
                                               "longitude":[-73.14],
                                               "depth":[150],
                                               "origin_time":[np.nan]})
    earthquakes = ttu.Earthquakes(earthquakes_data,"EPSG:3116")
    
    
    ymin, xmin = ttu.single_latlon2yx_in_km(-1.403,-81.819,"EPSG:3116")
    ymax, xmax= ttu.single_latlon2yx_in_km(14.571,-68.161,"EPSG:3116")
    x = (xmin,xmax)
    y = (ymin,ymax)
    z = (-5,200)
    P_profile = {"depth":[-10,4,25,32,40,100,200],
           "vel":[4.8,6.6,7,8,8.1,8.2,8.2]
           }
    S_profile = {"depth":[-10,4,25,32,40,100,200],
           "vel":[2.6966,3.7079,3.9326,4.4944,4.5505,4.6067,4.6067]
           }
    
    
    # ptt = EarthquakeTravelTime("P",stations,earthquakes)
    # ptt.add_grid_with_velocity_model(x,y,z,50,60,80,
    #                 xy_epsg="EPSG:3116",
    #                 vel1d=P_profile)
    # tt_path = "/home/emmanuel/ecastillo/dev/associator/GPA/data/tt/earthquakes.h5"
    # tt = ptt.get_traveltime(output=tt_path)
    # tt.plot(stations)
    
    file_paths = {"col":'/home/emmanuel/ecastillo/dev/associator/GPA/data/col.bna',
                    "map":'/home/emmanuel/ecastillo/dev/associator/GPA/data/map.bna',
                   "prv":'/home/emmanuel/ecastillo/dev/associator/GPA/data/prv.bna'}
    
    coordinates = { }
    for key,file_path in file_paths.items():
        coordinates[key] = []
        with open(file_path, 'r') as file:
            for line in file:
                lon, lat = map(float, line.strip().split(','))
                coordinates[key].append((lon, lat))
    
    polygons = list(coordinates.values())
    # print(polygons)
    # print(len(polygons))
    # exit()
    # polygons = []
    
    tt_path = "/home/emmanuel/ecastillo/dev/associator/GPA/data/tt/p_tt_dataset_COL.h5"
    ptt = EarthquakeTravelTimeDataset("P",stations,polygons=polygons)
    ptt.add_grid_with_velocity_model(x,y,z,50,60,80,
                    xy_epsg="EPSG:3116",
                    vel1d=P_profile)
    # ptt.save_traveltimes(output=tt_path)
    
    # tt_path = "/home/emmanuel/ecastillo/dev/associator/GPA/data/tt/s_tt_dataset_COL.h5"
    # stt = EarthquakeTravelTimeDataset("S",stations,polygons=polygons)
    # stt.add_grid_with_velocity_model(x,y,z,50,60,80,
    #                 xy_epsg="EPSG:3116",
    #                 vel1d=S_profile)
    # stt.save_traveltimes(output=tt_path)
    
    # n_event = 121150
    # tt = ptt.read_traveltimes_from_single_earthquake(tt_path,n_event,merge_stations=False)
    # tt.plot(stations)
    
    while True:
        n_event = int(random.uniform(0,130559))
        print(n_event)
        tt = ptt.read_traveltimes_from_single_earthquake(tt_path,n_event,merge_stations=False)
        # # print(tt.data)
        tt.plot(stations)
        
    # tt = ptt.get_traveltime()
    #     tt.plot(stations)
    
    
    # tt_path = "/home/emmanuel/ecastillo/dev/associator/GPA/data/tt/earthquakes.h5"
    # ptt = EarthquakeTravelTime("P",stations,earthquakes)
    # tt = ptt.read_traveltimes_from_single_earthquake(tt_path,0)
    # print(tt.data)
    # tt.plot(stations)
    