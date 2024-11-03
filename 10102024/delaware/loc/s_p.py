
from delaware.vel.vel import ScalarVelModel, ScalarVelPerturbationModel
import os
import logging
import datetime
from tqdm import tqdm
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from delaware.core.database import save_dataframe_to_sqlite,load_dataframe_from_sqlite
from delaware.core.eqviewer import Catalog,Picks
import numpy as np


logger = logging.getLogger("delaware.loc.s_p")

def get_std_deviations(v_mean,vps_min,vps_max):
    
    sigma_p = (v_mean/2)*((vps_max-vps_min)/(vps_max+vps_min)) 
    sigma_s = (v_mean/4)*((vps_max-vps_min)/(vps_max*vps_min))
    
    return sigma_p,sigma_s 

class SP_MontecarloSetup():
    def __init__(self,eqpicks,
                 stations,
                 vel_model,
                 z_guess,
                 output_folder,
                 vps_min =1.4,
                 vps_max = 2,
                 n_perturbations=1000,
                 scale_factor=1,
                 configure_logging=True):
        
        self.eqpicks = eqpicks
        self.stations = stations
        self.vel_model = vel_model
        self.z_guess = z_guess
        self.scale_factor = scale_factor
        self.vps_min = vps_min
        self.vps_max = vps_max
        self.n_perturbations = n_perturbations
        self.d = z_guess/(10**scale_factor)
        
        self.output_folder = output_folder
        
        folder_paths = ["vel","catalog","log"]
        self.folder_paths = {x:os.path.join(self.output_folder,
                                            self.eqpicks.author,
                                            x) for x in \
                                folder_paths
                             }
        for value in self.folder_paths.values():
            if not os.path.isdir(value):
                os.makedirs(value)
        
        if configure_logging:
            self._setup_logging()
            
    def _setup_logging(self):
        """
        Set up logging configuration.

        """
        if not os.path.isdir(self.folder_paths["log"]):
            os.makedirs(self.folder_paths["log"])
            
        log_file = os.path.join(self.folder_paths["log"],"print.log")
        

        timenow = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.logging_path = log_file

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
                
    def prepare_params(self):
        content = f"reference_vel_model:{self.vel_model.name}\n"
        content = f"folder_paths:{self.folder_paths}\n"
        content += f"z_guess:{self.z_guess}\n"
        content += f"d:{self.d}\n"
        content += f"vps_min:{self.vps_min}\n"
        content += f"vps_max:{self.vps_max}\n"
        content += f"n_perturbations:{self.n_perturbations}"
        
        log_file = os.path.join(self.folder_paths["log"],"params.txt")
        if not os.path.isdir(self.folder_paths["log"]):
            os.makedirs(self.folder_paths["log"])
            
        logger.info(f"Preparing parameters:{log_file}")
        
        with open(log_file, "w") as file:
            # Write the content to the file
            file.write(content)
        
    def prepare_catalog(self):
        logger.info(f"Preparing catalog:{self.folder_paths['catalog']}")
        catalog, phases = self.stations.get_events_by_sp(catalog=self.eqpicks.catalog,
                                                         picks_path=self.eqpicks.picks_path,
                                                         rmax=self.d,
                                                         zmin=self.z_guess,
                                                         output_folder=self.folder_paths["catalog"]
                                                         )
        # print(catalog,phases)
        
    def prepare_velocities(self):
        logger.info(f"Preparing velocities:{self.folder_paths['vel']}")
        
        
        vel_model_output = os.path.join(self.folder_paths["vel"],self.vel_model.name+".png")
        self.vel_model.plot_profile(savefig=vel_model_output,show=False)
        
        vp = self.vel_model.get_average_velocity("P",zmax=self.z_guess)
        vs = self.vel_model.get_average_velocity("S",zmax=self.z_guess)
        
        sigma_p,sigma_s = get_std_deviations(vp,vps_min=self.vps_min,
                                             vps_max=self.vps_max)
        
        svm = ScalarVelModel(p_value=vp,s_value=vs,
                             name=self.vel_model.name)
        
        svm.get_perturbation_model(
                                   output_folder=self.folder_paths["vel"],
                                   n_perturbations=self.n_perturbations,
                                   p_std_dev=sigma_p,
                                   s_std_dev=sigma_s,
                                   log_file = True,
                                   )
        
    def run(self):
        
        self.prepare_params()
        self.prepare_velocities()
        self.prepare_catalog()
    
class SP_Montecarlo():
    def __init__(self,root,depth,author,xy_epsg):
        
        folder_paths = ["vel","catalog","log"]
        self.folder_paths = {x:os.path.join(root,f"z_{depth}",
                                            author,
                                            x) for x in \
                                folder_paths
                             }
        
        catalog_path = os.path.join(self.folder_paths["catalog"],"catalog_sp_method.db")
        picks_path = os.path.join(self.folder_paths["catalog"],"picks_sp_method.db")
        vel_path = os.path.join(self.folder_paths["vel"],"perturbations.npz")
        
        catalog = load_dataframe_from_sqlite(db_name=catalog_path)
        picks = load_dataframe_from_sqlite(db_name=picks_path)
        
        picks["arrival_time"] = pd.to_datetime(picks["arrival_time"])
        
        self.catalog = Catalog(catalog,xy_epsg=xy_epsg)
        self.picks = Picks(picks,author=author)
        self.vel = ScalarVelPerturbationModel(vel_path)
    
    @property
    def n_stations(self):
        stations = self.picks.drop_duplicates("station")
        n_stations = len(stations)
        return n_stations
    
    @property
    def n_events(self):
        events = self.picks.drop_duplicates("ev_id")
        n_events = len(events)
        return n_events
    
    def __str__(self) -> str:
        
        msg = f"Stations | {self.n_stations} stations, {self.n_events} events "
        return msg
    
    def run_single_montecarlo_analysis(self,z_guess,output_folder=None):
        """Considering the suggestion of uses guesses of z (alexandros idea)

        Args:
            z_guess (_type_): _description_
            min_vps (float, optional): _description_. Defaults to 1.5.
            max_vps (float, optional): _description_. Defaults to 1.8.
            output_folder (_type_, optional): _description_. Defaults to None.
        """
        
        # Initialize tqdm for pandas
        # for n,event in tqdm(self.catalog.iterrows(),
        #                     total=len(self.catalog),
        #                     desc="Events"):
        for n,event in self.catalog.data.iterrows():
            
            ev_id = event["ev_id"]
            picks_by_id = self.picks.data.query(f"ev_id == '{ev_id}'")
            
            if picks_by_id.empty:
                print(f"No picks in event {ev_id}")
                continue
            # print(picks_by_id.info)
            p_phase = picks_by_id.query(f"phase_hint == 'P'") 
            s_phase = picks_by_id.query(f"phase_hint == 'S'") 
            sp_time = s_phase.iloc[0].arrival_time - p_phase.iloc[0].arrival_time
            sp_time = sp_time.total_seconds()
            station = p_phase.iloc[0].station
            
            
            vp_disp = round((max_p_vel -min_p_vel)/2,2)
            vs_disp = round((max_s_vel -min_s_vel)/2,2)

            vp = min_p_vel+vp_disp
            vs = min_s_vel+vs_disp 
            
            z = (sp_time)*min_p_vel/(min_vps-1)
            # zz = (sp_time)*min_p_vel/(min_vps-1)
            # zzz = (sp_time)*max_p_vel/(max_vps-1)
            print(sp_time,z)
            # print(vp,vp_disp)
            # print(vs,vs_disp )
            # print(zz,zzz)
            # exit()
            
            # data = {"z":[],"vp":[],"vs":[]}
            # for i in range(len(scalar_vel_perturbation)):
            #     vp = scalar_vel_perturbation.p_vel[i]
            #     vs = scalar_vel_perturbation.s_vel[i]
            #     vps =  vp/vs
            #     z = (vp/vps)*sp_time
            #     data["z"].append(z)
            #     data["vp"].append(vp)
            #     data["vs"].append(vs)
                
            # data = pd.DataFrame(data)
                
            # # Insert the model_id column to track each perturbation model
            # data.insert(0, "ev_id", ev_id)
            # data["station"] = station
            # data["ts-tp"] = sp_time
            # data["original_z"] = event["depth"]
                
            # save_dataframe_to_sqlite(data, output, table_name=ev_id)
        
    def run_montecarlo(self,scalar_vel_perturbation,output):
        
        """
        Based on a scalar velocity perturbation done in advance.
        First results presented on Castillo2025_102024
        """
        
        # Initialize tqdm for pandas
        for n,event in tqdm(self.catalog.iterrows(),
                            total=len(self.catalog),
                            desc="Events"):
            
            ev_id = event["id"]
            picks_by_id = self.picks.query(f"ev_id == '{ev_id}'")
            
            if picks_by_id.empty:
                print(f"No picks in event {ev_id}")
                continue
            
            p_phase = picks_by_id.query(f"phase_hint == 'P'") 
            s_phase = picks_by_id.query(f"phase_hint == 'S'") 
            sp_time = s_phase.iloc[0].time - p_phase.iloc[0].time
            sp_time = sp_time.total_seconds()
            station = p_phase.iloc[0].station_code
            
            data = {"z":[],"vp":[],"vs":[]}
            for i in range(len(scalar_vel_perturbation)):
                vp = scalar_vel_perturbation.p_vel[i]
                vs = scalar_vel_perturbation.s_vel[i]
                vps =  vp/vs
                z = (vp/vps)*sp_time
                data["z"].append(z)
                data["vp"].append(vp)
                data["vs"].append(vs)
                
            data = pd.DataFrame(data)
                
            # Insert the model_id column to track each perturbation model
            data.insert(0, "ev_id", ev_id)
            data["station"] = station
            data["ts-tp"] = sp_time
            data["original_z"] = event["depth"]
                
            save_dataframe_to_sqlite(data, output, table_name=ev_id)
                
        
    def plot_stations_counts(self):
        
        
        data = self.picks.copy()
        data = data.drop_duplicates("ev_id")
        
        # First, count the number of occurrences of each station_name
        station_counts = data.groupby('station_code')['ev_id'].count()

        # Plot the histogram
        fig,ax = plt.subplots(1,1,figsize=(10, 6))
        # plt.figure(figsize=(10, 6))
        station_counts.plot(kind='bar',color="coral")
        
        for idx, value in enumerate(station_counts):
            ax.text(idx, value, f'{value}', ha='center', va='bottom', fontsize=14)
        
        ax.set_title('S-P Method\nNumber of events per station',
                     fontdict={"size":18})
        ax.set_xlabel('Stations',fontdict={"size":14})
        ax.set_ylabel('Events',fontdict={"size":14})
        
        # Add the total number of ev_ids in a text box
        total_ev_ids = station_counts.sum()
        text_str = f"Total events: {total_ev_ids}"
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.5))
        
        ax.grid(True, linestyle='--', alpha=0.7) 
        
        plt.xticks(rotation=90, fontsize=14)  # Rotate x labels for readability
        plt.yticks(fontsize=14)  # Rotate x labels for readability
        plt.tight_layout()  # Adjust layout to avoid cutting off labels
        plt.show()