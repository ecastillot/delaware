# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-09-23 16:03:02
#  * @modify date 2024-09-23 16:03:02
#  * @desc [description]
#  */
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from delaware.core.database import save_dataframe_to_sqlite,load_dataframe_from_sqlite

class VelModel():
    def __init__(self, data, name, dtm=None) -> None:
        self.name = name
        self.dtm = dtm
        
        if self.dtm is not None:
            data["Depth (km)"].replace("DTM", self.dtm, inplace=True)
            
        self.data = data.astype(float)
        
    def __str__(self, extended=False) -> str:
        if extended:
            msg = f"VelModel1D: {self.name}"
        else:
            msg = f"VelModel1D: {self.name}\n{self.data}"

        return msg

    def plot_profile(self, zmin: float = None, zmax: float = None,
                     show_vp=True,show_vs=True,
                     show: bool = True, savefig:str=None):
        """
        Plot velocity profile.

        Args:
            zmin (float): Minimum depth to plot. Default is None.
            zmax (float): Maximum depth to plot. Default is None.
            show (bool): Whether to show the plot. Default is True.
            savefig: File path to save the plot. Default is None.

        Returns:
            Figure: Matplotlib figure object.
        """
        fig = plt.figure()
        rect = fig.patch
        rect.set_facecolor('w')

        ax = fig.add_subplot(1, 1, 1)
        if show_vp:
            ax.step(self.data["VP (km/s)"], self.data["Depth (km)"], 'k', linewidth=2.5, linestyle='-', label='VP (km/s)')
        if show_vs:
            ax.step(self.data["VS (km/s)"], self.data["Depth (km)"], 'r', linewidth=2.5, linestyle='-', label='VS (km/s)')
        ax.legend(loc='lower left')

        ax.set_xlabel('Velocity [km/s]')
        ax.set_ylabel('Depth [km]')
        ax.set_ylim(ymin=zmin, ymax=zmax)
        ax.invert_yaxis()
        ax.grid()
        
        if savefig is not None:
            fig.savefig(savefig)
        
        if show:
            plt.show()
        
        return fig,ax

    def _get_perturbations(self, n=1000, p_mean=0, p_std_dev=0.05,
                               s_mean=0, s_std_dev=0.05):
        """
        Perform Monte Carlo simulation by generating multiple realizations of the velocity model.
        
        Parameters:
        - n: Number of perturbations
        - mean: Mean of Gaussian noise for each perturbation
        - std_dev: Standard deviation of Gaussian noise for each perturbation
        
        Returns:
        - A dictionary of results for VP and VS perturbations, each containing arrays of velocities per depth.
        """
        data = self.data.copy()
        
        layers = self.data.shape[0]
        
        params = {"VP (km/s)":{"mean":p_mean,"std_dev":p_std_dev},
                  "VS (km/s)":{"mean":s_mean,"std_dev":s_std_dev}}
        
        pert_vels = {"VP (km/s)":{},"VS (km/s)":{}}
        for i in range(layers):
            for vel_key in ["VP (km/s)","VS (km/s)"]:
                noise = np.random.normal(params[vel_key]["mean"], 
                                         params[vel_key]["std_dev"],
                                         n)
                pert_vels[vel_key][i] = data.iloc[i][vel_key] + noise
        
        return pert_vels
    
    def monte_carlo_simulation(self, output,num_simulations=1000, p_mean=0, p_std_dev=0.05,
                               s_mean=0, s_std_dev=0.05):
        """
        Perform Monte Carlo simulation by generating multiple realizations of the velocity model.
        
        Parameters:
        - num_simulations: Number of simulations to run
        - mean: Mean of Gaussian noise for each simulation
        - std_dev: Standard deviation of Gaussian noise for each simulation
        
        Returns:
        - A dictionary of results for VP and VS simulations, each containing arrays of velocities per depth.
        """
        
        
        perturbations = self._get_perturbations(n=num_simulations,p_mean=p_mean,
                                                s_mean=s_mean,
                                                p_std_dev=p_std_dev,
                                                s_std_dev=s_std_dev)
        
        # print(perturbations["VP (km/s)"])
        # exit()
        # print(list(perturbations["VP (km/s)"].keys()))
        # print(list(perturbations["VP (km/s)"][0].keys()))
        # exit()        
        for i in tqdm(range(num_simulations),"Perturbations"):
            
            data = self.data.copy()
            data.insert(0, "model_id", i)
            
            # print(f"Perturbation {i}")
            
            vels_by_simulation = {"VP (km/s)":[],"VS (km/s)":[]}
            for wave_key in vels_by_simulation.keys():
                vel_by_phase = perturbations[wave_key]
                
                for layer,vel in vel_by_phase.items():
                    vels_by_simulation[wave_key].append(vel[i])
                    
                    
            vels_by_simulation = pd.DataFrame.from_dict(vels_by_simulation)
            
            # new_cols = ["Delta "+key for key in list(vels_by_simulation.keys())]
            # renaming = dict(zip(vels_by_simulation.keys(),new_cols))
            
            # print(data)
            # vels_by_simulation.rename(columns=renaming,inplace=True)
            data["Delta VP (km/s)"] = data["VP (km/s)"] - vels_by_simulation ["VP (km/s)"]
            data["Delta VS (km/s)"] = data["VS (km/s)"] - vels_by_simulation ["VS (km/s)"]
            data["VP (km/s)"] = vels_by_simulation ["VP (km/s)"]
            data["VS (km/s)"] = vels_by_simulation ["VS (km/s)"]
            
            # print(data,"\n")
            save_dataframe_to_sqlite(data,output,table_name=f"model_{i}")
            # simulation = pd.concat([data[""],vels_by_simulation],axis=1)
            # print(data)
            # exit()
        
            
            
class MontecarloVelModel():
    def __init__(self,db_path,models=None) -> None:
        self.db_path = db_path
        
        if models is not None:
            models = np.arange(*models)
            models = [ f"model_{i}" for i in models]
            
        data = load_dataframe_from_sqlite(db_path,models)
        self.models = models
        self.data = data
    
    @property
    def values_by_depth(self):
        return self.data.groupby("Depth (km)")
    
    def plot(self,vel_model, n_bins=30,
                          columns=["VP (km/s)","VS (km/s)"],
                          colors=["k","r"],
                          zmin=None,
                          zmax=None,
                          show=True):
        
        
        fig,axes = plt.subplots(1,2)
        rect = fig.patch
        rect.set_facecolor('w')
        
        all_x = []

        for i,vel_name in enumerate(columns):
            
            vbd = self.values_by_depth
            for j in range(len(vel_model.data)-1):
                
                z0 = vel_model.data.loc[j,"Depth (km)"]
                z1 = vel_model.data.loc[j+1,"Depth (km)"]
                
                data = vbd.get_group(z0)
                
                print(data)
                
                counts, bins = np.histogram(data[columns[i]], bins=n_bins, density=False)
                
                norm_counts = (counts - counts.min()) / (counts.max() - counts.min())
                
                counts_z0 = norm_counts
                counts_z1 = norm_counts
                
                counts_2d = np.vstack([counts_z0, counts_z1])
                extent = [bins[0], bins[-1], z0, z1]  # Map x to bins and y to depths
                im = axes[i].imshow(counts_2d, cmap='viridis', aspect='auto', 
                                    extent=extent, origin='lower')
                
                all_x.extend(bins)
        
            # print(self.data[columns[i]], self.data["Depth (km)"])
            axes[i].step(vel_model.data[columns[i]], vel_model.data["Depth (km)"], colors[i],
                    linewidth=2.5, linestyle='-', label=vel_model.name)
        
            axes[i].set_xlabel('Velocity [km/s]')
            axes[i].set_ylabel('Depth [km]')
            axes[i].legend(loc="lower left")
            
            axes[i].set_ylim(ymin=zmin, ymax=zmax)
            axes[i].invert_yaxis()
            axes[i].grid()
            axes[i].set_title(columns[i])
        
        # Set common limits
        x_min, x_max = min(all_x), max(all_x)

        for ax in axes:
            ax.set_xlim([np.floor(x_min), np.floor(x_max)])
            ax.set_xticks(np.arange(np.floor(x_min), np.floor(x_max),1))  # Set same tick separation
        
        # Show the entire plot after the loop finishes
        plt.colorbar(im, ax=axes[i], label='PDF')
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return fig, axes
        # exit()
