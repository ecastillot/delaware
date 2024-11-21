
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
import seaborn as sns


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
                                                         output_folder=self.folder_paths["catalog"]
                #  zmin=self.z_guess, # it shouldn't be consider because we don't want to consider the catalog depth.
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
        self.depth = depth
        
        folder_paths = ["vel","catalog","log","montecarlo"]
        self.folder_paths = {x:os.path.join(root,f"z_{depth}",
                                            author,
                                            x) for x in \
                                folder_paths
                             }
        
        for value in self.folder_paths.values():
            if not os.path.isdir(value):
                os.makedirs(value)
        
        self.catalog_path = os.path.join(self.folder_paths["catalog"],"catalog_sp_method.db")
        self.picks_path = os.path.join(self.folder_paths["catalog"],"picks_sp_method.db")
        self.vel_path = os.path.join(self.folder_paths["vel"],"perturbations.npz")
        self.montecarlo_path = os.path.join(self.folder_paths["montecarlo"],"montecarlo.db")
        self.montecarlo_depth_path = os.path.join(self.folder_paths["montecarlo"],"montecarlo_depth.png")
        self.montecarlo_times_path = os.path.join(self.folder_paths["montecarlo"],"montecarlo_times.png")
        self.stations_count_path = os.path.join(self.folder_paths["montecarlo"],"stations_counts.png")
        
        catalog = load_dataframe_from_sqlite(db_name=self.catalog_path)
        picks = load_dataframe_from_sqlite(db_name=self.picks_path)
        
        picks["arrival_time"] = pd.to_datetime(picks["arrival_time"])
        
        self.catalog = Catalog(catalog,xy_epsg=xy_epsg)
        self.picks = Picks(picks,author=author)
        self.vel = ScalarVelPerturbationModel(self.vel_path)
    
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
    
    def run_montecarlo(self):
        
        """
        """
        title = f"Depth: {self.depth} km"
        
        self.plot_stations_counts(savefig=self.stations_count_path,
                                  title=title,
                                  show=False)
        
        picks = self.picks.copy()
        
        all_data = []
        # Initialize tqdm for pandas
        for n,event in tqdm(self.catalog.data.iterrows(),
                            total=len(self.catalog),
                            desc="Events"):
            
            ev_id = event["ev_id"]
            picks_by_id = picks.data.query(f"ev_id == '{ev_id}'")
            
            if picks_by_id.empty:
                print(f"No picks in event {ev_id}")
                continue
            
            p_phase = picks_by_id.query(f"phase_hint == 'P'") 
            s_phase = picks_by_id.query(f"phase_hint == 'S'") 
            sp_time = s_phase.iloc[0].arrival_time - p_phase.iloc[0].arrival_time
            sp_time = sp_time.total_seconds()
            station = p_phase.iloc[0].station
            
            data = {"z":[],"vp":[],"vs":[]}
            for i in range(len(self.vel)):
                vp = self.vel.p_vel[i]
                vs = self.vel.s_vel[i]
                vps =  vp/vs
                z = (vp/(vps-1))*sp_time
                data["z"].append(z)
                data["vp"].append(vp)
                data["vs"].append(vs)
                
            data = pd.DataFrame(data)
                
            # Insert the model_id column to track each perturbation model
            data.insert(0, "ev_id", ev_id)
            data["station"] = station
            data["ts-tp"] = sp_time
            data["original_z"] = event["depth"]
            
            # print(data)
            # exit()
            
            # print(station,data.describe()) 
            all_data.append(data)  
            save_dataframe_to_sqlite(data, self.montecarlo_path, table_name=ev_id)
        
        all_data = pd.concat(all_data)
        
        
        plot_montecarlo_depths_by_station(all_data,
                                          title= title,
                                          savefig=self.montecarlo_depth_path,
                                          show=False)
        plot_montecarlo_times_by_station(all_data,
                                          title= title,
                                         savefig=self.montecarlo_times_path,
                                          show=False)
                
        
    def plot_stations_counts(self, title=None,savefig:str=None,
             show=True):
        
        
        data = self.picks.data.copy()
        data = data.drop_duplicates("ev_id")
        
        # First, count the number of occurrences of each station_name
        station_counts = data.groupby('station')['ev_id'].count()

        # Plot the histogram
        fig,ax = plt.subplots(1,1,figsize=(10, 6))
        # plt.figure(figsize=(10, 6))
        station_counts.plot(kind='bar',color="coral")
        
        for idx, value in enumerate(station_counts):
            ax.text(idx, value, f'{value}', ha='center', va='bottom', fontsize=14)
        
        # ax.set_title('S-P Method\nNumber of events per station',
        #              fontdict={"size":18})
        if title is not None:
            title = f"Number of events per station\n{title}"
            ax.set_title(title,
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
        
        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
            
        return fig,ax

def plot_montecarlo_times_by_station(data,title=None,show: bool = True, savefig:str=None):   
    
    data = data.drop_duplicates("ev_id")
    # grouped = data.groupby('station')['ev_id']
    fig,ax = plt.subplots(1,1,figsize=(10, 6))
    sns.boxplot(data=data,x="station",y="ts-tp",ax=ax)
    
    # Calculate the number of samples for each station
    sample_counts = data['station'].value_counts()

    # Annotate the box plot with sample counts
    for station, count in sample_counts.items():
        # Get the x position of each station
        x_pos = data['station'].unique().tolist().index(station)
        # Set the annotation above each box plot
        ax.text(x_pos, data['ts-tp'].max(), f'n={count}', 
                ha='center', va='bottom', color='black')
    if title is not None:
        title = r'$t_{\mathrm{s}} - t_{\mathrm{p}}$' +f' Analysis\n{title}'
        ax.set_title(title,
                        fontdict={"size":18})
    ax.set_xlabel('Stations',fontdict={"size":14})
    ax.set_ylabel(r'$t_{\mathrm{s}} - t_{\mathrm{p}}$ (s)',fontdict={"size":14})
    plt.xticks(fontsize=14)  # Rotate x labels for readability
    plt.yticks(fontsize=14)  # Rotate x labels for readability
    plt.tight_layout()  # Adjust layout to avoid cutting off labels
        
    if savefig is not None:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig,ax
    
        
def plot_montecarlo_depths_by_station(data,title=None,show: bool = True, savefig:str=None):   
    
    # Assuming your DataFrame is named df
    grouped = data.groupby('station')

    fig, ax = plt.subplots(1, 1)
    
    # colors = ["blue","green","magenta","brown","black","cyan"]
    # Plot histograms for each ev_id
    for i,(station, group) in enumerate(grouped):
        zeros = np.zeros(len(group['original_z']))
        
        if i ==0:
            label = "Original Depths"
        else:
            label=None
            
        ax.plot(group['original_z'],zeros, alpha=0.7,
                marker="x", color="black", markersize=10,
                label=label)
        
        
        id_grouped = group.groupby('ev_id')
        
        for j,(ev_id, ev_group) in enumerate(id_grouped):
            if j ==0:
                sta_label = station
            else:
                sta_label=None
            # print(station,colors[i])
            ax.hist(ev_group['z'], 
                    # color=colors[i], 
                    bins=30, alpha=0.3,
                    # density=True,
                    density=False,
                     histtype='step',
                    label=sta_label)
        
        # ax.scatter(group['original_z'], zeros, alpha=0.7, 
        #           marker='x', s=100, color='red')
        
    if title is not None:
        title = f"Earthquake Depth Analysis\n{title}"
        ax.set_title(title,
                        fontdict={"size":18})
    ax.set_xlabel('z (km)',fontdict={"size":18})
    ax.set_ylabel('Frequency',fontdict={"size":18})
    ax.set_xlim(0,20)
    ax.legend()
    
    plt.xticks(fontsize=14)  # Rotate x labels for readability
    plt.yticks(fontsize=14)  # Rotate x labels for readability
    plt.tight_layout()  # Adjust layout to avoid cutting off labels
        
    if savefig is not None:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig,ax

def plot_times_by_station(data,title=None,show: bool = True, savefig:str=None,
                          **plot_kwargs):   
    
    data = data.drop_duplicates("ev_id")
    # grouped = data.groupby('station')['ev_id']
    fig,ax = plt.subplots(1,1,figsize=(10, 6))
    sns.boxplot(data=data,x="station",y="ts-tp",ax=ax,**plot_kwargs)
    
    # Calculate the number of samples for each station
    sample_counts = data['station'].value_counts()
    if "order" in plot_kwargs.keys():
        sample_counts = sample_counts.reindex(plot_kwargs["order"])
    print(sample_counts)

    # Annotate the box plot with sample counts
    for i,(station, count) in enumerate(sample_counts.items()):
        # Get the x position of each station
        # x_pos = sample_counts.unique().tolist().index(station)
        # Set the annotation above each box plot
        ax.text(i, data['ts-tp'].max()-(data['ts-tp'].max()*0.1),
                f'n={count}', 
                ha='center', va='bottom', color='black',
                fontsize=18)
    if title is not None:
        title = r'$t_{\mathrm{s}} - t_{\mathrm{p}}$' +f' Analysis\n{title}'
        ax.set_title(title,
                        fontdict={"size":16})
    ax.set_xlabel('Stations',fontdict={"size":14})
    ax.set_ylabel(r'$t_{\mathrm{s}} - t_{\mathrm{p}}$ (s)',fontdict={"size":14})
    plt.xticks(fontsize=14)  # Rotate x labels for readability
    plt.yticks(fontsize=14)  # Rotate x labels for readability
    plt.tight_layout()  # Adjust layout to avoid cutting off labels
        
    if savefig is not None:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig,ax