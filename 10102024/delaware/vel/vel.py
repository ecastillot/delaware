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
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from delaware.core.database import save_dataframe_to_sqlite,load_dataframe_from_sqlite

class ScalarVelModel():
    def __init__(self,p_value,s_value,name) -> None:
        self.name = name
        self.p_value = p_value
        self.s_value = s_value
    
    def get_perturbation_model(self,output_folder, n_perturbations=1000, 
                            p_std_dev=0.05,
                               s_std_dev=0.05,
                               log_file=True):
        
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
            
        output_file = os.path.join(output_folder,"perturbations.npz")
        log_file = os.path.join(output_folder,"log_perturbations.txt")
        
        p_vel = np.random.normal(self.p_value,p_std_dev,n_perturbations)
        s_vel = np.random.normal(self.s_value,s_std_dev,n_perturbations)
        
        
        if log_file:
            content = f"reference_vel_model:{self.name}\n"
            content = f"path:{output_file}\n"
            content += f"n_perturbations:{n_perturbations}\n"
            content += f"vp_mean:{self.p_value}\n"
            content += f"vs_mean:{self.s_value}\n"
            content += f"sigma_vp:{p_std_dev}\n"
            content += f"sigma_vs:{s_std_dev}"
            
            
            with open(log_file, "w") as file:
                # Write the content to the file
                file.write(content)
        
        np.savez(output_file,p_vel=p_vel,s_vel=s_vel)

class ScalarVelPerturbationModel:
    def __init__(self,npz_path) -> None:
        self.npz_path = npz_path
        data = np.load(npz_path)
        self.p_vel = data["p_vel"]     
        self.s_vel = data["s_vel"] 
    
    def __len__(self):
        return len(self.p_vel)
        
    def plot(self, n_bins=30,
             colors=["k", "r"], savefig:str=None,
             show=True):
        
        fig, ax = plt.subplots(1, 1)
        
        
        ax.hist(self.p_vel,color=colors[0], 
                label="VP (km/s)",
                density=False, bins=n_bins)  # density=False would make counts
        ax.hist(self.s_vel,color=colors[1], 
                label="VS (km/s)",
                density=False, bins=n_bins)  # density=False would make counts
        
        # Set axis labels and title
        ax.set_xlabel('Velocity [km/s]')
        ax.set_ylabel('Counts')
        ax.legend()  # Add legend to the plot
        ax.grid()  # Add grid lines to the plot
        ax.set_title("Velocity histograms")  # Set title of the subplot
        
        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
            
        return fig,ax


#used
class VelModel:
    """
    A class to represent a 1D velocity model.

    Attributes:
        name (str): Name of the velocity model.
        dtm (float): Depth To Model (DTM) value used to replace "DTM" in data.
        data (DataFrame): Data containing velocity information.

    Methods:
        __str__: Returns a string representation of the velocity model.
        plot_profile: Plots the velocity profile with various options.
    """

    def __init__(self, data, name, dtm=None) -> None:
        """
        Initialize the VelModel instance.

        Args:
            data (DataFrame): The data containing depth and velocity information.
            name (str): The name of the velocity model.
            dtm (float, optional): Replace 'DTM' reference for your actual depth value in km. Default is None.
        """
        self.name = name
        self.dtm = dtm
        
        self._mandatory_columns = ['Depth (km)','VP (km/s)','VS (km/s)']
        # Check if all mandatory columns are present in the DataFrame
        check = all(item in data.columns.to_list() for item in self._mandatory_columns)
        if not check:
            raise Exception("There is not the mandatory columns for the data in Earthquakes object." \
                            + f"->{self._mandatory_columns}")

        # Replace "DTM" with the specified depth model value, if provided.
        if self.dtm is not None:
            # data["Depth (km)"].replace({"DTM": self.dtm}, inplace=True)
            data.replace({"Depth (km)":{"DTM": self.dtm}}, inplace=True)
        
        # Convert data to float type for consistency in calculations.
        data = data.astype(float)
        data = data.sort_values('Depth (km)',ignore_index=True)
        self.data = data

    def __str__(self, extended=False) -> str:
        """
        Returns a string representation of the velocity model.

        Args:
            extended (bool, optional): If True, only display the model name.
                Otherwise, display both name and data. Default is False.

        Returns:
            str: Formatted string representing the velocity model.
        """
        
        if extended:
            msg = f"VelModel1D: {self.name}\n{self.data}"
        else:
            msg = f"VelModel1D: {self.name}"

        return msg

    def plot_profile(self, zmin: float = None, zmax: float = None,
                     show_vp=True,show_vs=True,
                     show: bool = True, 
                     savefig:str=None):
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
            ax.step(self.data["VP (km/s)"], 
                    self.data["Depth (km)"], 
                    'k', linewidth=2.5, linestyle='-', 
                    label='VP (km/s)')
        if show_vs:
            ax.step(self.data["VS (km/s)"], 
                    self.data["Depth (km)"], 'r', 
                    linewidth=2.5, linestyle='-', 
                    label='VS (km/s)')
        ax.legend(loc='lower left')

        ax.set_xlabel('Velocity [km/s]')
        ax.set_ylabel('Depth [km]')
        ax.set_ylim(ymin=zmin, ymax=zmax)
        ax.invert_yaxis()
        ax.grid()
        
        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
        
        return fig,ax

    def get_average_velocity(self, phase_hint, zmin=None, zmax=None):
        """
        Calculate the depth-weighted average velocity within a specified depth range,
        accounting for partial layers at zmin and zmax.

        Args:
            zmin (float, optional): Minimum depth for the range. Default is None.
            zmax (float, optional): Maximum depth for the range. Default is None.
            phase_hint (str, optional): Type of wave to calculate the average velocity for.
                Options are 'P' (P-wave velocity) and 'S' (S-wave velocity). 

        Returns:
            float: The depth-weighted average velocity within the specified depth range.
        """
        # Filter data based on zmin and zmax and sort by depth
        filtered_data = self.data.sort_values(by="Depth (km)").reset_index(drop=True)
        
        # Initialize variables to hold the weighted sum and total depth
        weighted_sum = 0.0
        total_depth = 0.0
        
        # Loop through depth intervals to calculate weighted velocities
        for i in range(len(filtered_data) - 1):
            depth_start = filtered_data["Depth (km)"][i]
            depth_end = filtered_data["Depth (km)"][i + 1]
            velocity = filtered_data[f"V{phase_hint} (km/s)"][i]
            
            # Determine the effective start and end of the interval within zmin/zmax
            interval_start = max(depth_start, zmin) if zmin is not None else depth_start
            interval_end = min(depth_end, zmax) if zmax is not None else depth_end
            
            # print(interval_start,interval_end)
            # Only consider intervals that fall within the zmin-zmax range
            if interval_start < interval_end:
                interval_depth = interval_end - interval_start
                weighted_sum += velocity * interval_depth
                total_depth += interval_depth
                # print(interval_depth,weighted_sum,total_depth,"\n")
                # print(interval_depth,velocity,total_depth,"\n")
        
        # Handle the case if zmax exceeds the last defined depth
        if zmax is not None and zmax > filtered_data["Depth (km)"].iloc[-1]:
            last_velocity = filtered_data[f"V{phase_hint} (km/s)"].iloc[-1]
            interval_depth = zmax - filtered_data["Depth (km)"].iloc[-1]
            weighted_sum += last_velocity * interval_depth
            total_depth += interval_depth
            
        v = weighted_sum / total_depth
        v = round(v,2)
        return v if total_depth > 0 else None



    # This method generates perturbations for VP and VS velocity values
    def _get_perturbations(self, n=1000, p_mean=0, p_std_dev=0.05,
                           s_mean=0, s_std_dev=0.05):
        """
        Generate perturbations for the velocity model.
        
        Args:
            n (int): Number of perturbations to generate for each velocity.
            p_mean (float): Mean of the perturbation noise for VP.
            p_std_dev (float): Standard deviation of the perturbation noise for VP.
            s_mean (float): Mean of the perturbation noise for VS.
            s_std_dev (float): Standard deviation of the perturbation noise for VS.
            
        Returns:
            dict: Dictionary with perturbed VP and VS values for each layer.
        """
        # Make a copy of the original data
        data = self.data.copy()
        
        # Get the number of layers from the data shape
        layers = self.data.shape[0]
        
        # Define the parameters for VP and VS perturbation (mean and standard deviation)
        params = {
            "VP (km/s)": {"mean": p_mean, "std_dev": p_std_dev},
            "VS (km/s)": {"mean": s_mean, "std_dev": s_std_dev}
        }
        
        # Initialize dictionaries to store perturbed velocity values
        pert_vels = {"VP (km/s)": {}, "VS (km/s)": {}}
        
        # Loop over each layer to apply perturbation
        for i in range(layers):
            # Loop over both VP and VS
            for vel_key in ["VP (km/s)", "VS (km/s)"]:
                # Generate random noise based on the normal distribution
                noise = np.random.normal(params[vel_key]["mean"],
                                         params[vel_key]["std_dev"],
                                         n)
                # Add the noise to the original velocity values
                pert_vels[vel_key][i] = data.iloc[i][vel_key] + noise
        
        return pert_vels
    
    # This method generates perturbation models and saves them to a database
    def get_perturbation_model(self, output, n_perturbations=1000, p_mean=0, p_std_dev=0.05,
                               s_mean=0, s_std_dev=0.05):
        """
        Create perturbation models and save them to the database.
        
        Args:
            output (str): The name of the output SQLite database.
            n_perturbations (int): Number of perturbation models to create.
            p_mean (float): Mean of the perturbation noise for VP.
            p_std_dev (float): Standard deviation of the perturbation noise for VP.
            s_mean (float): Mean of the perturbation noise for VS.
            s_std_dev (float): Standard deviation of the perturbation noise for VS.
        """
        # Generate perturbations for the given parameters
        perturbations = self._get_perturbations(n=n_perturbations,
                                                p_mean=p_mean,
                                                s_mean=s_mean,
                                                p_std_dev=p_std_dev,
                                                s_std_dev=s_std_dev)
        
        # Loop through each perturbation and apply it to the data
        for i in tqdm(range(n_perturbations), "Perturbations"):
            # Make a copy of the original data
            data = self.data.copy()
            
            # Insert the model_id column to track each perturbation model
            data.insert(0, "model_id", i)
            
            # Initialize dictionaries to store the velocity per simulation
            vels_by_simulation = {"VP (km/s)": [], "VS (km/s)": []}
            
            # Loop through VP and VS velocities
            for wave_key in vels_by_simulation.keys():
                vel_by_phase = perturbations[wave_key]
                
                # Loop through each layer and apply the i-th perturbation
                for layer, vel in vel_by_phase.items():
                    vels_by_simulation[wave_key].append(vel[i])
            
            # Convert the perturbed velocity dictionary to a DataFrame
            vels_by_simulation = pd.DataFrame.from_dict(vels_by_simulation)
            
            # Calculate the differences between the original and perturbed velocities
            data["Delta VP (km/s)"] = data["VP (km/s)"] - vels_by_simulation["VP (km/s)"]
            data["Delta VS (km/s)"] = data["VS (km/s)"] - vels_by_simulation["VS (km/s)"]
            
            # Update the VP and VS columns with the perturbed velocities
            data["VP (km/s)"] = vels_by_simulation["VP (km/s)"]
            data["VS (km/s)"] = vels_by_simulation["VS (km/s)"]
            
            # Save the perturbed model to an SQLite database
            save_dataframe_to_sqlite(data, output, table_name=f"model_{i}")
        
class VelPerturbationModel:
    """
    A class to represent a velocity perturbation model. This class provides methods
    to load velocity model data from a database, group it by depth, and visualize the perturbations.
    
    Attributes:
        db_path (str): Path to the SQLite database containing velocity models.
        models (list): List of model names loaded from the database.
        data (DataFrame): The loaded data from the SQLite database.
    """

    def __init__(self, db_path, models=None) -> None:
        """
        Initialize the VelPerturbationModel with a database path and optional model range.
        
        Args:
            db_path (str): Path to the SQLite database.
            models (tuple, optional): A tuple representing the range of models to load (start, end). 
                                      Defaults to None, which loads all models.
        """
        self.db_path = db_path

        # Load specific models if a range is provided
        if models is not None:
            models = np.arange(*models)
            models = [f"model_{i}" for i in models]

        # Load data from the SQLite database
        data = load_dataframe_from_sqlite(db_path, models)
        self.models = models
        self.data = data

    @property
    def values_by_depth(self):
        """
        Group the data by depth.
        
        Returns:
            DataFrameGroupBy: A pandas groupby object with data grouped by 'Depth (km)'.
        """
        return self.data.groupby("Depth (km)")

    def plot(self, vel_model, n_bins=30,
             columns=["VP (km/s)", "VS (km/s)"],
             colors=["k", "r"],
             zmin=None, zmax=None,
             show=True):
        """
        Plot 2D histograms of velocity perturbations with depth and overlaid velocity models.
        
        Args:
            vel_model (VelModel): The velocity model to compare against.
            n_bins (int, optional): Number of bins for the histogram. Defaults to 30.
            columns (list, optional): List of columns to plot. Defaults to ["VP (km/s)", "VS (km/s)"].
            colors (list, optional): List of colors for plotting. Defaults to ["k", "r"].
            zmin (float, optional): Minimum depth for the plot. Defaults to None.
            zmax (float, optional): Maximum depth for the plot. Defaults to None.
            show (bool, optional): Whether to display the plot immediately. Defaults to True.
        
        Returns:
            tuple: The matplotlib figure and axes objects.
        """
        # Create a figure with two subplots for VP and VS
        fig, axes = plt.subplots(1, 2)
        rect = fig.patch
        rect.set_facecolor('w')  # Set the background color of the figure

        all_x = []  # To store all x-values for setting common limits

        # Iterate over the velocity columns (e.g., VP, VS)
        for i, vel_name in enumerate(columns):
            vbd = self.values_by_depth  # Group data by depth
            
            # Loop over each depth layer to create 2D histograms
            for j in range(len(vel_model.data) - 1):
                z0 = vel_model.data.loc[j, "Depth (km)"]  # Depth at the start of the layer
                z1 = vel_model.data.loc[j + 1, "Depth (km)"]  # Depth at the end of the layer
                
                # Get data corresponding to the current depth
                data = vbd.get_group(z0)
                
                # Generate a histogram for the current velocity column
                counts, bins = np.histogram(data[columns[i]], bins=n_bins, density=False)
                
                # Normalize the counts for better visualization
                norm_counts = (counts - counts.min()) / (counts.max() - counts.min())
                
                # Prepare the 2D array for plotting the histogram with depth
                counts_z0 = norm_counts
                counts_z1 = norm_counts
                counts_2d = np.vstack([counts_z0, counts_z1])  # Stack for both depths
                
                # Define the extent of the image in the plot
                extent = [bins[0], bins[-1], z0, z1]
                
                # Plot the 2D histogram as an image
                im = axes[i].imshow(counts_2d, cmap='viridis', aspect='auto',
                                    extent=extent, origin='lower')
                
                all_x.extend(bins)  # Store x-values (velocity) for common axis limits

            # Overlay the velocity model on top of the 2D histogram
            axes[i].step(vel_model.data[columns[i]], vel_model.data["Depth (km)"], colors[i],
                         linewidth=2.5, linestyle='-', label=vel_model.name)
            
            # Set axis labels and title
            axes[i].set_xlabel('Velocity [km/s]')
            axes[i].set_ylabel('Depth [km]')
            axes[i].legend(loc="lower left")  # Add legend to the plot
            axes[i].set_ylim(ymin=zmin, ymax=zmax)  # Set depth limits
            axes[i].invert_yaxis()  # Invert y-axis to show depth increasing downwards
            axes[i].grid()  # Add grid lines to the plot
            axes[i].set_title(columns[i])  # Set title of the subplot

        # Set common x-axis limits for both subplots
        x_min, x_max = min(all_x), max(all_x)
        for ax in axes:
            ax.set_xlim([np.floor(x_min), np.floor(x_max)])  # Set x-axis limits
            ax.set_xticks(np.arange(np.floor(x_min), np.floor(x_max), 1))  # Set x-ticks

        # Add a colorbar for the 2D histogram
        plt.colorbar(im, ax=axes[i], label='PDF')  
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Show the plot if the show argument is True
        if show:
            plt.show()

        return fig, axes  # Return figure and axes for further customization
        # exit()
