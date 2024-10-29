# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-09-24 16:49:27
#  * @modify date 2024-09-24 16:49:27
#  * @desc [description]
#  */
from tqdm import tqdm
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from delaware.core.database import save_dataframe_to_sqlite,load_dataframe_from_sqlite
from delaware.eqviewer.eqviewer import Catalog,Picks
import numpy as np

class SP_Database():
    def __init__(self,catalog_path,picks_path):
        self.catalog = load_dataframe_from_sqlite(db_name=catalog_path)
        self.picks = load_dataframe_from_sqlite(db_name=picks_path)
        
        self.picks["time"] = pd.to_datetime(self.picks["time"])
        
    
    @property
    def n_stations(self):
        stations = self.picks.drop_duplicates("station_code")
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
    
    def run_montecarlo_analysis(self,z_guess,min_vps=1.5,
                                max_vps=1.8,output_folder=None):
        """Considering the suggestion of uses guesses of z (alexandros idea)

        Args:
            z_guess (_type_): _description_
            min_vps (float, optional): _description_. Defaults to 1.5.
            max_vps (float, optional): _description_. Defaults to 1.8.
            output_folder (_type_, optional): _description_. Defaults to None.
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
            
            print(sp_time)
            
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
        # print(self.catalog)
    
    
def plot_montecarlo_depths(path):   
    data = load_dataframe_from_sqlite(db_name=path)
    
    # Assuming your DataFrame is named df
    grouped = data.groupby('ev_id')

    fig, ax = plt.subplots(1, 1)
    # Plot histograms for each ev_id
    for i,(ev_id, group) in enumerate(grouped):
        ax.hist(group['z'], bins=20, alpha=0.7,density=False)
        
        zeros = np.zeros(len(group['original_z']))
        
        if i ==0:
            label = "Original Depths"
        else:
            label=None
            
        ax.plot(group['original_z'],zeros, alpha=0.7,
                marker="x", color="black", markersize=10,
                label=label)
        # ax.scatter(group['original_z'], zeros, alpha=0.7, 
        #           marker='x', s=100, color='red')
        
    ax.set_title(f"Histogram of z",fontdict={"size":18})
    ax.set_xlabel('z (km)',fontdict={"size":18})
    ax.set_ylabel('Frequency',fontdict={"size":18})
    ax.legend()
    
    plt.xticks(fontsize=14)  # Rotate x labels for readability
    plt.yticks(fontsize=14)  # Rotate x labels for readability
    plt.show()
    
def plot_montecarlo_depths_by_station(path,show: bool = True, savefig:str=None):   
    data = load_dataframe_from_sqlite(db_name=path)
    
    # Assuming your DataFrame is named df
    grouped = data.groupby('station')

    fig, ax = plt.subplots(1, 1)
    
    colors = ["blue","green","magenta","brown"]
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
            print(station,colors[i])
            ax.hist(ev_group['z'], color=colors[i], 
                    bins=30, alpha=0.3,
                    # density=True,
                    density=False,
                     histtype='step',
                    label=sta_label)
        
        # ax.scatter(group['original_z'], zeros, alpha=0.7, 
        #           marker='x', s=100, color='red')
        
    ax.set_title(f"Histogram of z",fontdict={"size":18})
    ax.set_xlabel('z (km)',fontdict={"size":18})
    ax.set_ylabel('Frequency',fontdict={"size":18})
    ax.legend()
    
    plt.xticks(fontsize=14)  # Rotate x labels for readability
    plt.yticks(fontsize=14)  # Rotate x labels for readability
    if savefig is not None:
        fig.savefig(savefig)
    
    if show:
        plt.show()
    
    return fig,ax


def get_picks(event_ids,picks_path):
    
    return load_dataframe_from_sqlite(db_name=picks_path,
                                      tables=event_ids)

def get_events(origin, picks_path, event_ids=None,
               starttime=None,endtime=None,
               region=None, agencies=None,
               region_from_src = None,
               ):
    

    origin = origin.rename(columns = {"mag":"magnitude"})
    catalog = Catalog(origin)
    
    catalog.filter("origin_time",starttime,endtime)
    
    if (region is not None) and (len(catalog) !=0):
        catalog.filter_region(region)
        
    if (region_from_src is not None) and (len(catalog) !=0):
        lat,lon, r_max, az_max =  region_from_src
        catalog.filter_by_r_az(latitude=lat,
                           longitude=lon,
                           r=r_max,
                           az=az_max)
        
    if (event_ids is not None) and (len(catalog) !=0):
        catalog.select_data({"id":event_ids})
        
    if (agencies is not None) and (len(catalog) !=0):
        catalog.select_data({"agency":agencies}) #agencies is a list
        
    if len(catalog) != 0:
        event_ids = catalog.data["id"].to_list()
        
        picks = get_picks(event_ids=event_ids,picks_path=picks_path)
        
    
    else :
        picks = pd.DataFrame(columns=["ev_id"])
    
    picks = Picks(picks)
    
    
    return catalog, picks
    