# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-02-06 19:26:21
#  * @modify date 2024-02-06 19:26:21
#  * @desc [description]
#  */
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures as cf
from obspy.geodetics.base import gps2dist_azimuth
from delaware.synthetic.tt import TravelTime,EarthquakeTravelTimeDataset,EarthquakeTravelTime
from delaware.core.eqviewer import Source,Stations,Catalog
from delaware.core.eqviewer_utils import get_distance_in_dataframe
from delaware.synthetic.utils import change_file_extension
from delaware.vel.pykonal import VelModel
from delaware.synthetic.eqscenario_utils import *


##### Scenario Classes

class EarthquakeScenario(object):
    def __init__(self,
                 p_dataset_path: str,
                 s_dataset_path: str,
                 stations_path: str,
                 xy_epsg: str,
                 window: float = 300) -> None:
        """
        Initialize an EarthquakeScenario object.

        Args:
            p_dataset_path (str): Path to the P-wave dataset.
            s_dataset_path (str): Path to the S-wave dataset.
            stations_path (str): Path to the stations dataset.
            xy_epsg (str): EPSG code specifying the coordinate reference system.
            window (float, optional): Time window. Defaults to 300.
        """
        self.p_dataset_path = p_dataset_path
        self.s_dataset_path = s_dataset_path
        self.stations_path = stations_path
        self.xy_epsg = xy_epsg
        self.csv_dataset_path = change_file_extension(p_dataset_path, ".csv")
        self.p_npz_dataset_path = change_file_extension(p_dataset_path, ".npz")
        self.s_npz_dataset_path = change_file_extension(s_dataset_path, ".npz")
        self.p_aftershocks_folder = os.path.join(os.path.dirname(p_dataset_path),
                                               "aftershocks")
        self.s_aftershocks_folder = os.path.join(os.path.dirname(s_dataset_path),
                                               "aftershocks")
        self.window = window
        self.info = {"earthquakes":[],
                     "aftershocks":[],
                     "noise":[]}

    @property
    def stations(self):
        """
        Get stations information from the provided dataset.

        Returns:
            ttu.Stations: Stations object containing station information.
        """
        df = pd.read_csv(self.stations_path)
        stations = Stations(data=df, xy_epsg=self.xy_epsg)
        return stations

    @property
    def events(self):
        """
        Get earthquake events information.

        Returns:
            dict: A dictionary containing earthquake and aftershock events.
        """
        return {"earthquakes":self.info["earthquakes"], 
                "aftershocks":self.info["aftershocks"]}

    @property
    def noise(self):
        """
        Get the noise information.

        Returns:
            list: A list containing the noise information.
        """
        return self.info["noise"]

    def add_earthquakes(self, n_events: int, 
                        min_n_p_phases: int=4,
                        min_n_s_phases: int=2,):
        """
        Add earthquake events to the scenario.

        Args:
            n_events (int): Number of earthquake events to add.
            min_n_p_phases : int
                Number of minimum P phases.
            min_n_s_phases : int
                Number of minimum S phases.
        """
        csv_data = pd.read_csv(self.csv_dataset_path)
        start = csv_data["event_index"].min()
        end = csv_data["event_index"].max()

        event_indexes = random.sample(range(start, end + 1), n_events)
        eq_p_dataset = EarthquakeTravelTimeDataset("P", stations=self.stations)
        eq_s_dataset = EarthquakeTravelTimeDataset("S", stations=self.stations)

        def _get_events(event_index):
            """
            Helper function to get earthquake events.

            Args:
                event_index (int): Index of the earthquake event.
            """
            p_tt = eq_p_dataset.read_traveltimes_from_single_earthquake(input=self.p_dataset_path,
                                                                         event_id=event_index,
                                                                         merge_stations=True,
                                                                         dropinf=True)
            s_tt = eq_s_dataset.read_traveltimes_from_single_earthquake(input=self.s_dataset_path,
                                                                         event_id=event_index,
                                                                         merge_stations=True,
                                                                         dropinf=True)
            
            p_tt.data["phase_hint"] = "P"
            s_tt.data["phase_hint"] = "S"
            p_tt = get_distance_in_dataframe(p_tt.data,"src_lat","src_lon",
                                                "latitude","longitude")
            s_tt = get_distance_in_dataframe(s_tt.data,"src_lat","src_lon",
                                                "latitude","longitude")
            
            if p_tt.empty and s_tt.empty:
                return None
            elif not p_tt.empty and not s_tt.empty:
                tt = pd.concat([p_tt,s_tt],ignore_index=True)
            elif s_tt.empty:
                tt = p_tt
            elif p_tt.empty:
                tt = s_tt
                
            
            # getting first phase
            tt.sort_values("travel_time",inplace=True,ignore_index=True)
            tt['first_ev_phase'] = (tt.index == 0).astype(int)
            
            tt["event_index"] = tt["event_index"].astype(int)
            tt["event_type"] = "earthquake"
            tt["origin_time"] = "earthquake"
            tt["mainshock_latitude"] = None
            tt["mainshock_longitude"] = None
            tt["mainshock_depth"] = None
            tt["p_path"] = self.p_dataset_path
            tt["s_path"] = self.s_dataset_path
            
            # Placing the event in a window
            tt = get_travel_time_in_window(tt,min_n_p_phases,
                                               min_n_s_phases,self.window)
            
            tt.sort_values(["r","phase_hint"],inplace=True,ignore_index=True)
            self.info["earthquakes"].append(tt)

        with cf.ThreadPoolExecutor() as executor:
            executor.map(_get_events, event_indexes)
            
        # for event_index in event_indexes:
        #     _get_events(event_index)

    def add_afterschocks(self, mainshock: Source, 
                         n_aftershocks: int, 
                         radius_km: float,
                         min_n_p_phases: int=4,
                        min_n_s_phases: int=2, ):
        """
        Add aftershocks to the scenario.

        Args:
            mainshock (Source): Mainshock source information.
            n_aftershocks (int): Number of aftershocks to generate.
            radius_km (float): Radius in kilometers within which to generate aftershocks.
            min_n_p_phases : int
                Number of minimum P phases.
            min_n_s_phases : int
                Number of minimum S phases.

        """
        random_points = generate_random_points(mainshock.latitude,
                                               mainshock.longitude,
                                               mainshock.depth, radius_km,
                                               n_aftershocks)
        df = pd.DataFrame(random_points, columns=["latitude", "longitude", "depth"])
        df["origin_time"] = np.nan
        earthquakes = Earthquakes(data=df, xy_epsg=mainshock.xy_epsg)

        ## aftershocks paths
        timestamp = time.time()
        time_struct = time.gmtime(timestamp)
        time_string = time.strftime("%Y%m%dT%H%M%S", time_struct)
        p_aftershocks_path = os.path.join(self.p_aftershocks_folder,
                                          f"p_{time_string}.h5") 
        s_aftershocks_path = os.path.join(self.s_aftershocks_folder,
                                          f"s_{time_string}.h5") 
        
        ## Loading the velocity models, and then computing the traveltimes
        eq_p = EarthquakeTravelTime(phase="P", stations=self.stations,
                                    earthquakes=earthquakes)
        eq_p.load_velocity_model(path=self.p_npz_dataset_path,
                                 xy_epsg=self.xy_epsg)
        p_tt = eq_p.get_traveltimes(merge_stations=True,
                                    output=p_aftershocks_path)
        p_tt.data.sort_values(by="event_index", inplace=True)

        eq_s = EarthquakeTravelTime(phase="S", stations=self.stations,
                                    earthquakes=earthquakes)
        eq_s.load_velocity_model(path=self.s_npz_dataset_path,
                                 xy_epsg=self.xy_epsg,)
        s_tt = eq_s.get_traveltimes(merge_stations=True,
                                    output=s_aftershocks_path)
        s_tt.data.sort_values(by="event_index", inplace=True)

        ## adding other info
        p_tt.data["phase_hint"] = "P"
        s_tt.data["phase_hint"] = "S"
        p_tt = get_distance_in_dataframe(p_tt.data,"src_lat","src_lon",
                                            "latitude","longitude")
        s_tt = get_distance_in_dataframe(s_tt.data,"src_lat","src_lon",
                                            "latitude","longitude")
        
        if p_tt.empty and s_tt.empty:
            return None
        elif not p_tt.empty and not s_tt.empty:
            all_tt = pd.concat([p_tt,s_tt],ignore_index=True)
        elif s_tt.empty:
            all_tt = p_tt
        elif p_tt.empty:
            all_tt = s_tt
            
        
        all_tt["event_index"] = all_tt["event_index"].astype(int)
        all_tt["event_type"] = "aftershock"
        all_tt["mainshock_latitude"] = mainshock.latitude
        all_tt["mainshock_longitude"] = mainshock.longitude
        all_tt["mainshock_depth"] = mainshock.depth
        all_tt["p_path"] = p_aftershocks_path
        all_tt["s_path"] = s_aftershocks_path
        
        # print(all_tt)
        # exit()
        
        # Placing the aftershocks in a window
        for event_index, single_event in all_tt.groupby("event_index"):
            
            # getting first phase
            single_event.sort_values("travel_time",inplace=True,ignore_index=True)
            single_event['first_ev_phase'] = (single_event.index == 0).astype(int)
            
            tt = get_travel_time_in_window(single_event,min_n_p_phases,
                                               min_n_s_phases,self.window)
            tt = tt.sort_values(["r","phase_hint"],ignore_index=True)
            self.info["aftershocks"].append(tt)

    def remove_phases(self,
                      max_hyp_distances:list = [120,200,500,1000],
                      probabilities:list = [0.6,0.25,0.1,0.05],
                      min_n_p_phases: int=4,
                        min_n_s_phases: int=2,
                      distance_weight:float=0.9,
                      dispersion_weight:float=0.1,
                      p_phase_factor:float=0.8):
        
        """
        Remove phases based on certain criteria.
        
        Criteria: 
        - For each phase, the hypocentral distance 'r' from station to source is computed.
        - 'r' is normalized with the value randomly obtained from 'max_hyp_distances'and 'probabilities'.
        - 'r_norm' > 1 will be removed. It means, it removes stations with hypocentral distances greater than the allowed.
        - The probability for each phase to be removed follows the next equation
        
          p(r_norm,d) = (w_r * r_norm + w_d * d)/ (w_r + w_d)
          
          where w_r is the distance weight, r_norm is the normalized hypocentral distance.
          w_d is the dispersion factor, d is a random value.
          
        - The equation shows that larger r_norm distances, higher prob to be removed
            In addition, sometimes the distance is not the only one factor to be considered.
            Adding a random value 'd' hopefully will cover other unexpected circunstances.
        - The remove process guarantee the minimum number of phases specified.

        Args:
            max_hyp_distances (list): Distances candidates to choose randomly the 
                                    maximum allowed hypocentral distance based on the probabilities parameter.
                                    i.e. if 120 is chosen, stations with distances greater than will be removed
            probabilities (list): Probability ranges too choose the maximum hypocentral distance to normalize.
            min_n_p_phases : int: Number of minimum P phases.
            min_n_s_phases : int  Number of minimum S phases.
            distance_weight (float): Weight for distance in the remove probability calculation.
            dispersion_weight (float): Weight for dispersion in the remove probability calculation.
            p_phase_factor (float): Factor for reducing the probability of removing P phases.
        Returns:
            dict: A dictionary containing earthquake and aftershock events with removed phases.
        """
        
        events = self.events
        
        self.info["earthquakes"] = [] 
        self.info["aftershocks"] = [] 
        
        for event_type,event in events.items():
            for single_event in event:
                
                ## THis is necessary in case of aftershocks
                ev = single_event.groupby("event_index")
                
                for event_index,data in ev:
                    
                    r_max = data["r"].abs().max()
                    r_perc25 = data["r"].abs().quantile(0.25)
                    
                    # last prob is divided in 2 to include the maximum distance
                    all_distances = max_hyp_distances + [r_max]
                    p = probabilities[0:-1] + [probabilities[-1]/2] + [probabilities[-1]/2] 
                    r_max_allowed = np.random.choice(all_distances,
                                                  p=p)
                    
                    
                    #guarantee of minimum specific minimum number of phases
                    len_n_phases = len(data[data["r"] <= r_max_allowed])
                    
                    min_n_phases = min_n_p_phases + min_n_s_phases
                    
                    if len_n_phases < min_n_phases:
                        r_max_allowed = r_perc25
                        
                    # print(r_max_allowed,len_n_phases)
                    
                    # Remove_prob column is going to be used to remove picks
                    # First it's based on the normalized distance from the station to the source.
                    # Greater distances have higher probabilities to be removed.
                    data["r_norm"] = data["r"] / r_max_allowed
                    data["remove_prob"] = data["r_norm"]
                    
                    # Adding a random value and then normalizing again
                    # The adittion of the random value is to add more dispersion. 
                    # Sometimes the distance is not the only one factor to remove phases.
                    # Adding the random value hopefully will cover other unexpected circunstances
                    # like phases not detected, station malfuction, etc.
                    data["remove_prob"] = (distance_weight*data["remove_prob"] +\
                                        dispersion_weight*np.random.rand(len(data)) ) \
                                        /(distance_weight+dispersion_weight)
                    
                    #guarantee of minimum specific minimum number of phases
                    p_data =data[data["phase_hint"]=="P"]
                    s_data =data[data["phase_hint"]=="S"]
                    p_data.loc[p_data["remove_prob"].rank() <= min_n_p_phases, "remove_prob"] = 0
                    s_data.loc[s_data["remove_prob"].rank() <= min_n_s_phases, "remove_prob"] = 0
                    data = pd.concat([p_data,s_data],ignore_index=True)
                    
                    
                    # If it's a P phase, reduce the probability controlled
                    # by the p_phase_factor parameter 
                    reducing_p_prob = lambda row: row['remove_prob']*p_phase_factor\
                                                if row['phase_hint'] == "P" \
                                                else row['remove_prob']
                    data["remove_prob"] = data.apply(reducing_p_prob , axis=1)
                    
                    # threshold_to_remove_phases 
                    threshold = np.random.rand(len(data))
                    
                    # Remove picks based on the calculated probability
                    to_remove = data[data["remove_prob"] > threshold].index
                    data.drop(to_remove, inplace=True)
                    
                    # Remove the temporary columns
                    data.drop(["r_norm", "remove_prob"], axis=1, inplace=True)
                    data.sort_values(["r","phase_hint"],inplace=True,ignore_index=True)
                    
                    # getting first phase again because some phases were removed
                    single_event.sort_values("travel_time",inplace=True,ignore_index=True)
                    single_event['first_ev_phase'] = (single_event.index == 0).astype(int)
                    
                    data.sort_values(["r","phase_hint"],inplace=True,ignore_index=True)
                    # p_phases = len(data[data["phase_hint"]=="P"])
                    # s_phases = len(data[data["phase_hint"]=="S"])
                    # info = {"ev_type":event_type,
                    #         "ev_id":event_index,
                    #         "r_max":r_max_allowed,
                    #         "# Phases": len(data),
                    #         "# P-Phases":p_phases,
                    #         "# S-Phases":s_phases
                    #         }
                    # print(info)
                    ## testing the result
                    # data_plot = data[data["phase_hint"]=="P"]
                    # tt = TravelTime(data_plot)
                    # stations = Stations(data_plot,xy_epsg=self.xy_epsg)
                    # tt.plot(stations)
                    
                    self.info[event_type].append(data)
                    
    def add_general_noise(self, n_phases: int, stations: list = None):
        """
        Add general noise to the seismic data.

        Args:
            n_phases (int): The number of noise phases to add.
            stations (list, optional): A list of stations to consider for adding noise. If None, all stations are considered.
        """
        # Make a copy of the data
        noise = self.stations.data.copy()

        # Filter the data if stations are provided
        if stations is not None:
            noise = noise[noise["station"] in stations]

        # Sample n_phases from the data
        noise = noise.sample(n_phases, replace=True,ignore_index=True)

        # Generate random travel times
        random_floats = [random.uniform(0, self.window) for _ in range(len(noise))]

        # Assign travel times to the noise data
        noise["travel_time"] = random_floats
        noise["window_travel_time"] = noise["travel_time"]
        
        noise["event_index"] = noise.index
        noise["event_type"] = "general_noise"
        noise["phase_hint"] = np.random.choice(['P', 'S'], size=len(noise))

        noise["src_lat"] = noise["latitude"]
        noise["src_lon"] = noise["longitude"]
        noise['src_x[km]'] = noise["x[km]"]
        noise['src_y[km]'] = noise["y[km]"]
        noise["src_z[km]"] = noise["elevation"]*-1
        noise["first_ev_phase"] = 1 #noise is the first and unique noise phase
        
        # Append the noise data to the info dictionary
        self.info["noise"].append(noise)
    
    def add_noise_to_the_station(self, n_phases:int, n_stations: int):
        """
        Add noise to the seismic data for specific stations.

        Args:
            n_phases (int): The number of noise phases to add for each station.
            n_stations (int): The number of stations to add noise to.
        """
        # Make a copy of the data
        noise = self.stations.data.copy()

        # Sample n_stations from the data
        noise = noise.sample(n_stations, replace=True, ignore_index=True)

        # Duplicate the sampled stations n_phases times
        noise = pd.concat([noise]*n_phases, ignore_index=True)

        # Generate random travel times
        random_floats = [random.uniform(0, self.window) for _ in range(len(noise))]

        # Assign travel times to the noise data
        noise["travel_time"] = random_floats
        noise["window_travel_time"] = noise["travel_time"]

        noise["event_index"] = noise.index
        noise["event_type"] = "noise2station"
        noise["phase_hint"] = np.random.choice(['P', 'S'], size=len(noise))

        noise["src_lat"] = noise["latitude"]
        noise["src_lon"] = noise["longitude"]
        noise['src_x[km]'] = noise["x[km]"]
        noise['src_y[km]'] = noise["y[km]"]
        noise["src_z[km]"] = noise["elevation"]*-1
        noise["first_ev_phase"] = 1 #noise is the first and unique noise phase

        # Sort the noise data by station
        noise.sort_values("station", ignore_index=True, inplace=True)

        # Append the noise data to the info dictionary
        self.info["noise"].append(noise)
    
    def get_phases(self):
        
        all_events = self.events
        all_noise = self.noise
        
        phases = []
        
        for events in all_events.values():
            for event in events:
                phases.append(event)
                
        for noise in all_noise:    
            phases.append(noise)
        
        if not phases:
            return pd.DataFrame
        
        phases= pd.concat(phases,ignore_index=True)
        
        columns = ['event_index','event_type', 'src_lat', 'src_lon', 'src_z[km]', 'src_x[km]',
                'src_y[km]','mainshock_latitude', 'mainshock_longitude',
                'mainshock_depth','station_index','network', 'station', 'longitude', 'latitude',
                'elevation', 'x[km]', 'y[km]', 'z[km]','r', 'az', 'baz',
                'phase_hint', 'first_ev_phase','travel_time','window_travel_time','p_path', 's_path']
        
        non_cols = [x for x in columns if x not in phases.columns.tolist()]
        for x in non_cols:
            phases[x] = None
        # print(non_cols)
        
        # phases['event_index'] = phases['event_index'].astype(int)
        phases = phases[columns]
        
        phases = phases.sort_values("window_travel_time",
                                    ignore_index=True)
        return phases
    
    def plot_phases(self,sort_by_source:Source=None,
                    show:bool = True):
        
        phases = self.get_phases()
        stations = self.stations
        
        return plot_eqscenario_phases(phases=phases,stations=stations,
                               sort_by_source=sort_by_source,
                               show=show)
        
class EarthquakeScenarioDataset(object):
    def __init__(self,
                 tt_dataset_paths: dict,
                 stations_path: str,
                 xy_epsg: str,
                 remove_phases_args: dict,
                 window: float = 300,
                 earthquakes_args: dict = {
                     "range": (1, 20),  # Range of possible numbers of earthquakes
                     "priority_factor": 5,  # Factor for weighting the number of earthquakes
                     "min_n_p_phases": 4,  # Minimum number of P phases per earthquake
                     "min_n_s_phases": 2  # Minimum number of S phases per earthquake
                 },
                 aftershocks_args: dict = {
                     "range": (1, 50),  # Range of possible numbers of aftershocks
                     "priority_factor": 5,  # Factor for weighting the number of aftershocks
                     "mainshock_polygons": [],  # Polygons defining mainshock epicenters for aftershocks
                     "radius_km": 20,  # Radius in kilometers for generating aftershocks around mainshocks
                     "min_n_p_phases": 4,  # Minimum number of P phases per aftershock
                     "min_n_s_phases": 2  # Minimum number of S phases per aftershock,
                 },
                 general_noise_args: dict = {  
                     "range": (1, 100)  # Range of possible numbers of general noise phases
                 },
                 noise2station_args: dict = {  
                     "range": (1, 100),  # Range of possible numbers of noise phases
                     "stations_range": (1, 5),  # Range of possible numbers of stations affected by noise
                     "priority_stations_factor": 80  # Factor for weighting the number of affected stations
                 }
                 ):
        """
        Initialize the EarthquakeScenarioDataset object.

        Parameters:
        - tt_dataset_paths (dict): Dictionary containing paths to dataset files.
        - stations_path (str): Path to stations file.
        - xy_epsg (str): EPSG code for xy coordinates.
        - remove_phases_args (dict): Arguments for removing phases.
        - window (float): Time window in seconds (default 300).
        - earthquakes_args (dict): Arguments for generating earthquakes (default values provided).
        - aftershocks_args (dict): Arguments for generating aftershocks (default values provided).
        - general_noise_args (dict): Arguments for adding general noise (default values provided).
        - noise2station_args (dict): Arguments for adding noise to stations (default values provided).
        
        
        The priority_factor in earthquake_args,aftershock_args and noise2station_args,
        is a parameter that influences how the weights decrease in the range specified. 
        A higher priority_factor leads to a faster decrease in weights as the element index increases.
        
        Example: range(1,6) & priority_factor = 2
        for the list: [1,2,3,4,5], the weights would be: [8, 6, 4, 2, 0], it's more likely to get the value 1.
        """
        # Initialize instance variables
        self.tt_dataset_paths = tt_dataset_paths
        self.stations_path = stations_path
        self.xy_epsg = xy_epsg
        self.remove_phases_args = remove_phases_args
        self.window = window
        self.earthquakes_args = earthquakes_args
        self.aftershocks_args = aftershocks_args
        self.general_noise_args = general_noise_args
        self.noise2station_args = noise2station_args

    def get_single_scenario(self,earthquakes=True,aftershocks=False,
                            general_noise=True,noise2station=True):
        """
        Generate a single earthquake scenario.

        Returns:
        - EarthquakeScenario: An instance of EarthquakeScenario representing the generated scenario.
        """
        # Generate an EarthquakeScenario instance
        eq_scenario = EarthquakeScenario(p_dataset_path=self.tt_dataset_paths["P"],
                                               s_dataset_path=self.tt_dataset_paths["S"],
                                               stations_path=self.stations_path,
                                               xy_epsg=self.xy_epsg,
                                               window=self.window)
        # Add earthquakes if specified
        if earthquakes:
            possible_n_events = list(range(*self.earthquakes_args["range"]))
            weights = generate_weights(len(possible_n_events),
                                           priority_factor=self.earthquakes_args["priority_factor"])
            n_events = random.choices(possible_n_events, weights=weights, k=1)[0]
            eq_scenario.add_earthquakes(n_events=n_events,
                                        min_n_p_phases=self.earthquakes_args["min_n_p_phases"],
                                        min_n_s_phases=self.earthquakes_args["min_n_s_phases"])

        # Add aftershocks if specified
        if aftershocks:
            # print("aftershocks",aftershocks)
            possible_n_events = list(range(*self.aftershocks_args["range"]))
            weights = generate_weights(len(possible_n_events),
                                           priority_factor=self.aftershocks_args["priority_factor"])
            n_events = random.choices(possible_n_events, weights=weights, k=1)[0]

            model_path = change_file_extension(self.tt_dataset_paths["P"], ".npz")
            model = VelModel.load_npz(model_path, "P", self.xy_epsg)
            points = model.filter_geo_coords(self.aftershocks_args["mainshock_polygons"])
            possible_coords = [(points["src_lon"].min(), points["src_lon"].max()),
                                (points["src_lat"].min(), points["src_lat"].max()),
                                (0, 40)
                                ]
            lon, lat, z = tuple(random.uniform(min_val, max_val) \
                                for min_val, max_val in possible_coords)
            mainshock = Source(latitude=lat, longitude=lon, depth=z,
                               xy_epsg=self.xy_epsg)
            # print("mainshock",mainshock)
            eq_scenario.add_afterschocks(mainshock=mainshock,
                                         n_aftershocks=n_events,
                                         radius_km=self.aftershocks_args["radius_km"],
                                         min_n_p_phases=self.aftershocks_args["min_n_p_phases"],
                                         min_n_s_phases=self.aftershocks_args["min_n_s_phases"])

        # Remove phases if specified
        eq_scenario.remove_phases(**self.remove_phases_args)

        # Add general noise if specified
        if general_noise:
            n_phases = random.randint(*self.general_noise_args["range"])
            eq_scenario.add_general_noise(n_phases=n_phases)

        # Add noise to station if specified
        if noise2station:
            n_phases = random.randint(*self.noise2station_args["range"])
            possible_n_stations = list(range(*self.noise2station_args["stations_range"]))
            weights = generate_weights(len(possible_n_stations),
                                           priority_factor=self.noise2station_args["priority_stations_factor"])
            n_stations = random.choices(possible_n_stations, weights=weights, k=1)[0]
            eq_scenario.add_noise_to_the_station(n_phases=n_phases,
                                                 n_stations=n_stations)

        return eq_scenario
    
    
        # print(eq_scenario.noise[0]["station"].drop_duplicates().to_list())
        # print(eq_scenario.noise[0]["phase_hint"].drop_duplicates().to_list())
        # print(n_phases,n_stations)
        # eq_scenario.plot_phases()
        # exit()
        # eq_scenario.plot_phases(sort_by_source=mainshock)
        # print( n_events)
        # print( n_events)
        # print( eq_scenario.events)

    def create_dataset(self,
                       start_sample:int,
                       end_sample:int,
                       output_folder:str,
                       earthquakes_perc=0.9,
                       aftershocks_perc=0.3,
                       noise2station_perc = 0.3
                       ):
        
        samples = end_sample - start_sample
        samples_with_earthquakes = round(samples*earthquakes_perc)
        samples_with_aftershocks = round(samples*aftershocks_perc)
        samples_with_noise2station = round(samples*noise2station_perc)

        def create_array_with_m_ones(n, m):
            arr = np.full(n, False)
            arr[np.random.choice(n, m, replace=False)] = True
            return arr
        
        earthquakes_array = create_array_with_m_ones(samples,samples_with_earthquakes)
        aftershocks_array = create_array_with_m_ones(samples,samples_with_aftershocks)
        noise2station_array = create_array_with_m_ones(samples,samples_with_noise2station)
        # print(samples)
        # print(earthquakes_array,len(earthquakes_array))
        # print(aftershocks_array,len(aftershocks_array))
        # print(noise2station_array ,len(noise2station_array ))
        # exit()
        
        
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        
        # print(start_sample,end_sample)   
        for i in range(start_sample,end_sample):
                       
            filepath = calculate_filepath(i,output_folder=output_folder,
                                          file_format="csv")
            
            # window_index = i-1
            window_index = i-start_sample
            info = (
                    f"Window {i} | earthquakes:{earthquakes_array[window_index]}"
                    f" aftershocks:{aftershocks_array[window_index]}"
                    f" general_noise:{True}"
                    f" noise2station:{noise2station_array[window_index]}"
                    )
            print(info)
            scenario = self.get_single_scenario(earthquakes=earthquakes_array[window_index],
                                                aftershocks=aftershocks_array[window_index],
                                                general_noise=True,
                                                noise2station=noise2station_array[window_index])
            
            
            phases = scenario.get_phases()
            
            # columns = ['event_index','event_type', 'src_lat', 'src_lon', 'src_z[km]', 'src_x[km]',
            #     'src_y[km]','mainshock_latitude', 'mainshock_longitude',
            #     'mainshock_depth','station_index','network', 'station', 'longitude', 'latitude',
            #     'elevation', 'x[km]', 'y[km]', 'z[km]','r', 'az', 'baz',
            #     'phase_hint', 'travel_time','window_travel_time','p_path', 's_path']
            # phases = phases[columns]
            
            phases.to_csv(filepath,index=False)
            
            
            print("\t"+filepath)
            # print(exit())
            
            # print(phases.columns)
            # print(phases)
            
            
            # scenario.plot_phases()
            
        # print(earthquakes_array,aftershocks_array,general_noise_array,
        #       noise2station_array)
        
        
        # i = 1
        # while i <= samples:
            
if __name__ == "__main__":
    p_dataset_path = "/home/emmanuel/ecastillo/dev/associator/GPA/data/tt/p_tt_dataset_COL.h5"
    s_dataset_path = "/home/emmanuel/ecastillo/dev/associator/GPA/data/tt/s_tt_dataset_COL.h5"
    stations_path = "/home/emmanuel/ecastillo/dev/associator/GPA/data/CM_stations.csv"
    
    ##polygons
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
    
    tt_dataset_paths = {"P":p_dataset_path,"S":s_dataset_path}
    
    epd = EarthquakeScenarioDataset(tt_dataset_paths=tt_dataset_paths,
                           stations_path=stations_path,
                           xy_epsg="EPSG:3116",
                           remove_phases_args={
                               "max_hyp_distances": [100,200,250,350,500,1000],
                                "probabilities": [0.2,0.4,0.2,0.1,0.05,0.05],
                                "min_n_p_phases": 4,
                                "min_n_s_phases": 2,
                                "distance_weight":0.9,
                                "dispersion_weight":0.1,
                                "p_phase_factor":0.8
                                            },
                           window=300,
                            earthquakes_args={
                                                    "range":(1,5),
                                                    "priority_factor":30,
                                                    "min_n_p_phases":4,
                                                    "min_n_s_phases":2},
                            aftershocks_args={
                                                    "range":(3,20),
                                                    "priority_factor":5,
                                                    "mainshock_polygons":polygons,
                                                    "radius_km":20,
                                                    "min_n_p_phases":4,
                                                    "min_n_s_phases":2},
                            general_noise_args={
                                                    "range":(1,70)
                                                    },
                            noise2station_args={
                                                    "range":(1,25),
                                                    "stations_range":(1,3),
                                                    "priority_stations_factor":5,
                                                    }
                           )
    while True:
        scenario = epd.get_single_scenario(aftershocks=True)
        scenario.plot_phases()
        plt.show()
    
    # output_folder = "/home/emmanuel/ecastillo/dev/associator/GPA/data/simple_dataset/raw"
    # scenario = epd.create_dataset(start_sample=0,
    #                               end_sample=100,
    #                               output_folder=output_folder,
    #                             #   earthquakes_perc=0.1,
    #                             #   aftershocks_perc=0.9
    #                               )
    
    # output_folder = "/home/emmanuel/ecastillo/dev/associator/GPA/data/events"
    # scenario = epd.create_dataset(10,output_folder=output_folder)
    
    ## single scenario
    
    # eq_scenario = EarthquakeScenario(p_dataset_path=p_dataset_path,
    #                                  s_dataset_path=s_dataset_path,
    #                                  stations_path=stations_path,
    #                                  xy_epsg="EPSG:3116")
    # eq_scenario.add_earthquakes(4)
    # mainshock = Source(latitude=4.4160,
    #                 longitude=-73.8894,
    #                 depth=20,xy_epsg="EPSG:3116")
    # eq_scenario.add_afterschocks(mainshock=mainshock,
    #                              n_aftershocks=5,
    #                              radius_km=20)
    # eq_scenario.remove_phases()
    # # # print(eq_scenario.events)
    # # # print(len(eq_scenario.events["earthquakes"]))
    # eq_scenario.add_general_noise(100)
    # eq_scenario.add_noise_to_the_station(n_phases=40,n_stations=2)
    
    # # # eq_scenario.get_phases()
    # eq_scenario.plot_phases()
    # # # eq_scenario.plot_phases(sort_by_source=mainshock)
    # # print(eq_scenario.info)