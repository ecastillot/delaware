# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-09-23 16:03:02
#  * @modify date 2024-09-23 16:03:02
#  * @desc [description]
#  */
import numpy as np
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

    def _add_gaussian_perturbation(self, column_name, mean=0, std_dev=0.05):
        """
        Add Gaussian noise to each layer in a DataFrame column.
        
        Parameters:
        - column_name: Column in DataFrame to modify (e.g., 'VP (km/s)')
        - mean: Mean of the Gaussian noise distribution
        - std_dev: Standard deviation of the Gaussian noise distribution
        """
        data = self.data.copy()
        noise = np.random.normal(mean, std_dev, self.data.shape[0])
        data[column_name] += noise
        return data[column_name]

    def monte_carlo_simulation(self, output,num_simulations=1000, mean=0, std_dev=0.05):
        """
        Perform Monte Carlo simulation by generating multiple realizations of the velocity model.
        
        Parameters:
        - num_simulations: Number of simulations to run
        - mean: Mean of Gaussian noise for each simulation
        - std_dev: Standard deviation of Gaussian noise for each simulation
        
        Returns:
        - A dictionary of results for VP and VS simulations, each containing arrays of velocities per depth.
        """
        data = self.data.copy()
        data.insert(0, "model_id", 0)
        
        for i in range(num_simulations):
            data["model_id"] = i
            data["VP (km/s)"] = self._add_gaussian_perturbation("VP (km/s)",mean, std_dev)
            data["VS (km/s)"] = self._add_gaussian_perturbation("VS (km/s)",mean, std_dev)
            data[" Delta_VP (km/s)"] = self.data["VP (km/s)"] - data["VP (km/s)"]
            data[" Delta_VS (km/s)"] = self.data["VS (km/s)"] - data["VS (km/s)"]
            
            save_dataframe_to_sqlite(data,output,table_name=f"model_{i}")
        
            
            
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

        for i,vel_name in enumerate(columns):
            
            vbd = self.values_by_depth
            for j in range(len(vel_model.data)-1):
                
                z0 = vel_model.data.loc[j,"Depth (km)"]
                z1 = vel_model.data.loc[j+1,"Depth (km)"]
                
                data = vbd.get_group(z0)
                
                counts, bins = np.histogram(data[columns[i]], bins=n_bins, density=False)
                
                norm_counts = (counts - counts.min()) / (counts.max() - counts.min())
                
                counts_z0 = norm_counts
                counts_z1 = norm_counts
                
                counts_2d = np.vstack([counts_z0, counts_z1])
                extent = [bins[0], bins[-1], z0, z1]  # Map x to bins and y to depths
                im = axes[i].imshow(counts_2d, cmap='viridis', aspect='auto', 
                                    extent=extent, origin='lower')
        
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
            
        
            # Show the entire plot after the loop finishes
        plt.colorbar(im, ax=axes[i], label='PDF')
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return fig, axes
        # exit()
