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

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.step(self.data["VP (km/s)"], self.data["Depth (km)"], 'k', linewidth=2.5, linestyle='-', label='VP (km/s)')
        ax1.step(self.data["VS (km/s)"], self.data["Depth (km)"], 'r', linewidth=2.5, linestyle='-', label='VS (km/s)')
        ax1.legend(loc='lower left')

        ax1.set_xlabel('Velocity [km/s]')
        ax1.set_ylabel('Depth [km]')
        ax1.set_ylim(ymin=zmin, ymax=zmax)
        ax1.invert_yaxis()
        ax1.grid()
        
        if savefig is not None:
            fig.savefig(savefig)
        
        if show:
            plt.show()
        
        return fig

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
            
            save_dataframe_to_sqlite(data,output,table_name=f"model_{i}")
        
            
            
class MontecarloVelModel():
    def __init__(self,path) -> None:
        self.path = path
        data = load_dataframe_from_sqlite(path)
        print(data)
    
    # def load(self):
        

    # def plot_monte_carlo(self, num_simulations=1000, mean=0, std_dev=0.05):
    #     """
    #     Plot the velocity distributions for VP and VS using the Monte Carlo simulation results.
        
    #     Parameters:
    #     - num_simulations: Number of simulations to run
    #     - mean: Mean of Gaussian noise for each simulation
    #     - std_dev: Standard deviation of Gaussian noise for each simulation
    #     """
    #     simulations = self.monte_carlo_simulation(num_simulations=num_simulations, mean=mean, std_dev=std_dev)
        
    #     print(simulations["VP (km/s)"].shape)
        # exit()
        # depth = self.data['Depth (km)']
        
        # # Plotting the Vp and Vs results
        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 10), sharey=True)
        # cmap = plt.get_cmap('jet')
        
        
        
        # # Heatmap for Vp
        # num_iterations = len(simulations["vp"])
        # hist_vp, xedges_vp, yedges_vp = np.histogram2d(
        #     simulations["vp"].flatten(), np.repeat(depth, num_iterations),
        #     bins=[100, len(depth)], density=True
        # )
        # ax1.imshow(
        #     hist_vp.T, extent=[simulations["vp"].min(), simulations["vp"].max(),
        #                        depth.max(), depth.min()],
        #     aspect='auto', cmap=cmap, origin='upper'
        # )
        # ax1.set_title('Vp')
        # ax1.set_xlabel('Velocity (km/s)')
        # ax1.set_ylabel('Depth (km)')
        
        # # Add colorbars
        # fig.colorbar(ax1.imshow(hist_vp.T, 
        #                         extent=[simulations["vp"].min(), 
        #                                 simulations["vp"].max(), depth.max(), 
        #                         depth.min()],
        #                         aspect='auto', cmap=cmap, origin='upper'),
        #              ax=ax1, orientation='vertical', label='Probability Density')

        # plt.tight_layout()
        # plt.show()
        

