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



