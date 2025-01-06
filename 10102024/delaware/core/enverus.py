import pandas as pd
import glob
import os
import lasio
import numpy as np
import matplotlib.pyplot as plt
from delaware.vel.vel import VelModel
from matplotlib import cm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import time
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options



class Client:
    """
    A client for logging into Enverus and downloading well log data.
    """

    def __init__(self, user, pss, link=None, hide_driver=False):
        """
        Initializes the Client object with user credentials and configurations.

        Args:
            user (str): Username for authentication.
            pss (str): Password for authentication.
            link (str, optional): Login URL. Defaults to Enverus login URL.
            hide_driver (bool, optional): Whether to hide the browser driver. Defaults to False.
        """
        self.user = user
        self.pss = pss
        self.hide_driver = hide_driver

        if link is None:
            link = rf'https://login.auth.enverus.com/'
        
        self.link = link

    def query(self, log_df, url_fmt=None, download_folder=None):
        """
        Logs into the Enverus platform and downloads well log data.

        Args:
            log_df (DataFrame): A pandas DataFrame containing WellID and LogId columns.
            url_fmt (str, optional): URL format for accessing logs. Defaults to a standard format.
            download_folder (str, optional): Folder to save the downloaded files. If None, uses default folder.

        Raises:
            Exception: If any critical element or page fails to load.
        """
        # Set default URL format if not provided
        if url_fmt is None:
            url_fmt = "https://prism.enverus.com/prism/well/{well_id}/wellLog/{log_id}"
        
        # Remove duplicate rows based on WellID and LogId
        log_df = log_df.drop_duplicates(subset=["WellID", "LogId"], ignore_index=True)

        # Configure Chrome options if a download folder is specified
        if download_folder is not None:
            if not os.path.isdir(download_folder):
                os.makedirs(download_folder)
            
            chrome_options = Options()
            prefs = {
                "download.default_directory": download_folder,  # Set default download directory
                "download.prompt_for_download": False,         # Disable the download prompt
                "directory_upgrade": True,                     # Ensure the download folder exists
            }
            chrome_options.add_experimental_option("prefs", prefs)
        else:
            chrome_options = None

        # Initialize the WebDriver
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_window_size(1800, 1200)  # Ensure correct window size for screenshots

        # Navigate to the login page
        driver.get(self.link)

        # Enter username
        username_xpath = '//*[@id="auth0"]/div/div/form/div/div/div[3]/span/div/div/div/div/div/div/div/div/div/div[1]/div/input'
        input_field = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.XPATH, username_xpath))
        )
        input_field.send_keys(self.user)

        # Enter password
        password_xpath = '//*[@id="auth0"]/div/div/form/div/div/div[3]/span/div/div/div/div/div/div/div/div/div/div[2]/div[1]/div/input'
        input_field = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.XPATH, password_xpath))
        )
        input_field.send_keys(self.pss)

        # Click the login button
        login_button_xpath = '//*[@id="auth0"]/div/div/form/div/div/button'
        driver.find_element(By.XPATH, login_button_xpath).click()

        # Wait for the dashboard to load and click on the required section
        dashboard_xpath = '//*[@id="app-tray-section"]/di-carousel/div/div[1]'
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, dashboard_xpath))
        )
        element.click()

        # Wait for a new tab to open and switch to it
        WebDriverWait(driver, 10).until(lambda d: len(d.window_handles) > 1)
        new_tab = driver.window_handles[1]  # The second tab (index starts at 0)
        driver.switch_to.window(new_tab)

        # Iterate through the DataFrame rows and download logs
        for i, row in log_df.iterrows():
            url = url_fmt.format(well_id=row["WellID"], log_id=row["LogId"])
            driver.get(url)
            print(f"Downloaded: {url}")

        # Keep the browser open until the user manually closes it
        input("Press Enter to close the browser...")

        # Close the WebDriver
        driver.quit()
        

def get_well_names(folder):
    """
    Extract unique well names from LAS files in a given folder.

    Parameters:
        folder (str): Path to the directory containing LAS files.

    Returns:
        list: A list of unique well names extracted from the file names.
    """
    # Get all LAS file paths in the specified folder
    paths = glob.glob(os.path.join(folder, "*.LAS"))
    
    # Initialize an empty list to store well names
    well_names = []
    
    # Iterate through each LAS file path
    for path in paths:
        # Get the file name without extension
        basename = os.path.splitext(os.path.basename(path))[0]
        
        # Extract the well name (assumes it's the first part of the file name before an underscore)
        well_name = basename.split("_")[0]
        
        # Append the well name to the list
        well_names.append(well_name)
    
    # Remove duplicates by converting the list to a set and back to a list
    well_names = list(set(well_names))
    
    return well_names

def get_uwi12(las_path):
    """
    Extracts a 12-digit UWI (Unique Well Identifier) from the given LAS file path.

    Args:
        las_path (str): The full path to the LAS file.

    Returns:
        str: A 12-digit UWI extracted from the file name.
    """
    # Extract the base name of the file (e.g., "123456789012_ABC.LAS")
    basename = os.path.basename(las_path)
    
    # Extract the well name by splitting the base name at underscores and taking the first part
    well_name = basename.split("_")[0]
    
    # Remove the last three characters from the well name to get the 12-digit UWI
    uwi12 = well_name[:-3]
    
    return uwi12
    

def get_sonic_data(las_path, formation):
    """
    Extracts sonic data (compressional and shear wave velocities) and computes 
    true vertical depth (TVD) from a LAS file, aligning it with formation data.

    Args:
        las_path (str): Path to the LAS file.
        formation (pd.DataFrame): A DataFrame containing formation data with columns
                                  'API_UWI_12', 'MD_Top', and 'TVD_Top'.

    Returns:
        pd.DataFrame: A DataFrame with measured depth (MD), true vertical depth (TVD),
                      and calculated velocities (Vp and Vs) in km/s.

    Raises:
        Exception: If depth data or sonic data (DT or DT_S) is missing.
    """
    # Read the LAS file and convert it to a DataFrame
    las = lasio.read(las_path)
    df = las.df().reset_index()
    
    # Extract the 12-digit UWI from the LAS file name
    uwi12 = get_uwi12(las_path)
    
    # Filter formation data for the specific well (using UWI) and sort by MD_Top
    formation = formation[formation["API_UWI_12"] == uwi12]
    formation = formation.sort_values("MD_Top")
    
    # Extract the depth (MD) column from the LAS file
    md_values = None
    for x in ["DEPTH", "DEPT", "DEP"]:
        if x in df.columns.to_list():
            md_values = df[x]
            break  # Stop the loop once the depth column is found
    
    # Raise an exception if no depth column is found
    if md_values is None:
        raise Exception("No Depth column found in the LAS file")
    
    # Identify available sonic data columns (DT or DT_S)
    dt_columns = []
    if "DT" in df.columns.to_list():
        dt_columns.append("DT")
    if "DT_S" in df.columns.to_list():
        dt_columns.append("DT_S")
    
    # Raise an exception if no sonic data columns are found
    if not dt_columns:
        raise Exception(f"No DT or DT_S column found in LAS file for well {uwi12}")
    
    # Interpolate TVD values using MD and formation data
    md_real_values = [md_values.min()] + formation["MD_Top"].to_list() + [md_values.max()]
    tvd_real_values = [md_values.min()] + formation["TVD_Top"].to_list() + [md_values.max()]
    tvd_values = np.interp(md_values, md_real_values, tvd_real_values)
    
    # Create a DataFrame with MD and TVD, and add sonic data columns
    result = pd.DataFrame({"MD": md_values, "TVD": tvd_values})
    result = pd.concat([result, df[dt_columns]], axis=1)
    
    # Drop rows with missing sonic data
    result.dropna(subset=dt_columns, inplace=True)

    # Convert MD and TVD from feet to kilometers
    result["MD[km]"] = result["MD"] * 0.0003048
    result["TVD[km]"] = result["TVD"] * 0.0003048

    # Convert DT (µs/ft) to velocity (Vp and Vs) in km/s
    for dt in dt_columns:
        if dt == "DT":
            result["Vp[km/s]"] = 1 / (result["DT"] * 1e-6 * 3280.84)  # Compressional wave velocity
        elif dt == "DT_S":
            result["Vs[km/s]"] = 1 / (result["DT_S"] * 1e-6 * 3280.84)  # Shear wave velocity
    
    # Drop rows with missing velocity data
    result.dropna(subset=dt_columns, inplace=True)
    
    # Raise an exception if the resulting DataFrame is empty
    if result.empty:
        raise Exception("No valid Vp or Vs data available")
    
    # Return the processed DataFrame
    return result
    
def get_sonics_data(las_folder, formation, well):
    """
    Processes sonic data from multiple LAS files in a specified folder,
    aligning the data with formation and well information.

    Args:
        las_folder (str): Path to the folder containing LAS files.
        formation (pd.DataFrame): DataFrame containing formation data with
                                  'API_UWI_12', 'MD_Top', and 'TVD_Top' columns.
        well (pd.DataFrame): DataFrame containing well metadata with
                             'API_UWI_12', 'Latitude', 'Longitude', and 'ENVElevationGL_FT' columns.

    Returns:
        pd.DataFrame: A concatenated DataFrame containing processed sonic data for all wells,
                      with columns for well name, latitude, longitude, elevation, and sonic data.

    Raises:
        None: Exceptions during processing are caught and logged; problematic files are skipped.
    """
    # Get all LAS file paths in the specified folder
    paths = glob.glob(os.path.join(las_folder, "*.LAS"))

    # Dictionary to store processed data for each well
    wells = {}

    # Iterate through each LAS file in the folder
    for las_path in paths:
        # Extract the file name without extension
        basename = os.path.splitext(os.path.basename(las_path))[0]

        # Extract the well name (assumes the first part of the file name before an underscore)
        well_name = basename.split("_")[0]

        # Get the 12-digit UWI for the current LAS file
        uwi12 = get_uwi12(las_path)

        # Filter the well metadata for the current UWI
        single_well = well[well["API_UWI_12"] == uwi12]
        if single_well.empty:
            continue  # Skip if no matching well data is found

        # Extract well metadata (latitude, longitude, and elevation)
        single_well = single_well.iloc[0]
        latitude = single_well["Latitude"]
        longitude = single_well["Longitude"]
        elevation = single_well["ENVElevationGL_FT"] * 0.0003048  # Convert elevation from feet to kilometers

        try:
            # Process the LAS file to extract sonic data
            df = get_sonic_data(las_path, formation)
        except Exception as e:
            # Skip the LAS file if any error occurs
            continue

        # Add well metadata to the sonic data DataFrame
        df.insert(0, "well_name", well_name)
        df.insert(1, "latitude", latitude)
        df.insert(2, "longitude", longitude)
        df.insert(3, "elevation[km]", elevation)

        # Log successful processing of the well
        if well_name not in wells:
            print(f"{well_name} OK\t Latitude: {latitude}, Longitude: {longitude}, Elevation: {elevation:.3f} km")

        # Store the processed data in the wells dictionary
        wells[well_name] = df

    # Concatenate all well DataFrames into a single DataFrame
    wells = pd.concat(list(wells.values()))

    return wells

def get_dw_models():
    """
    Loads and initializes two velocity models from CSV files: Sheng and DB1D.

    Returns:
        tuple: A tuple containing two VelModel instances:
            - db1d_velmodel: The velocity model loaded from 'DB_model.csv'.
            - sheng_velmodel: The velocity model loaded from 'DW_model.csv'.
    """
    # Define the base directory path for the velocity model data
    dw_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/vel"

    # Load the Sheng velocity model
    vel_path = os.path.join(dw_path, "DW_model.csv")  # Path to Sheng model CSV file
    data = pd.read_csv(vel_path)  # Read Sheng model data into a DataFrame
    sheng_velmodel = VelModel(data=data, dtm=-0.7, name="Sheng")  # Initialize Sheng velocity model

    # Load the DB1D velocity model
    vel_path = os.path.join(dw_path, "DB_model.csv")  # Path to DB1D model CSV file
    data = pd.read_csv(vel_path)  # Read DB1D model data into a DataFrame
    db1d_velmodel = VelModel(data=data, dtm=-0.7, name="db1d")  # Initialize DB1D velocity model

    # Return the two velocity model objects
    return db1d_velmodel, sheng_velmodel

def plot_all_velocity_logs(
    data,
    formations=None,
    wells=None,
    depth="Depth[km]",
    vel="Vp[km/s]",
    smooth="moving_average",
    xlims=None,
    ylims=None,
):
    """
    Plots all velocity logs for wells, optionally applying smoothing 
    and overlaying reference velocity models.

    Args:
        data (DataFrame): Well data containing depth and velocity information.
        formations (DataFrame, optional): Formation tops with TVD and formation names.
        wells (list, optional): List of well names to include in the plot. Default is all wells.
        depth (str, optional): Column name for depth. Default is "Depth[km]".
        vel (str, optional): Column name for velocity. Default is "Vp[km/s]".
        smooth (str, optional): Smoothing method to apply. Default is "moving_average".
        xlims (tuple, optional): Tuple specifying x-axis limits (min, max).
        ylims (tuple, optional): Tuple specifying y-axis limits (min, max).

    Returns:
        None: Displays the velocity log plots.
    """
    # Group data by well name
    grouped_data = data.groupby("well_name").__iter__()

    # Use all wells if no specific wells are provided
    if wells is None:
        wells = list(set(data["well_name"].tolist()))

    # Create subplots for raw and smoothed data if smoothing is enabled
    if smooth is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))  # Adjusted figure size

    # Load reference velocity models
    db1d_velmodel, sheng_velmodel = get_dw_models()

    # Iterate over wells and plot velocity logs
    for well_name, single_data in grouped_data:
        # Sort data by depth for consistent plotting
        single_data = single_data.sort_values("TVD[km]")

        # Skip wells not in the specified list
        if well_name not in wells:
            continue

        # Extract formations for the current well
        api12 = well_name[0:-3]
        single_formation = formations[formations["API_UWI_12"] == api12]
        single_formation = single_formation.drop_duplicates("FormationName", ignore_index=True)

        # Adjust depth to account for elevation if necessary
        if depth == "Depth[km]":
            single_data[depth] = single_data["TVD[km]"] - single_data["elevation[km]"]

        # Apply smoothing if specified
        if smooth == "moving_average":
            single_data["Vp_smoothed"] = single_data["Vp[km/s]"].rolling(window=1000, center=True).mean()

        # Plot raw and smoothed velocity logs
        for i in range(len(axes)):
            if i == 0:
                axes[i].step(single_data[vel], single_data[depth], label=well_name)
            elif i == 1:
                axes[i].step(single_data["Vp_smoothed"], single_data[depth], label=well_name)

        # Plot formation tops for the well
        if not single_formation.empty:
            print(single_data["elevation[km]"].iloc[0])
            for row, f in single_formation.iterrows():
                depth_f = (f["TVD_Top"] * 0.0003048) - single_data["elevation[km]"].iloc[0]
                label_f = f["FormationName"]
                axes[i].axhline(y=depth_f, color="k", linestyle="--", linewidth=0.5)
                axes[i].text(
                    1.02,
                    depth_f,
                    label_f,
                    color="black",
                    fontsize=8,
                    ha="left",
                    va="top",
                    transform=axes[i].get_yaxis_transform(),
                )  # Label

    # Add reference velocity models and adjust axes
    for ax in axes:
        ax.step(
            db1d_velmodel.data["VP (km/s)"],
            db1d_velmodel.data["Depth (km)"],
            "blue",
            linewidth=2.5,
            linestyle="-",
            label="DB1D",
        )
        ax.step(
            sheng_velmodel.data["VP (km/s)"],
            sheng_velmodel.data["Depth (km)"],
            "orange",
            linewidth=2.5,
            linestyle="-",
            label="Sheng (2022)",
        )
        # Set axis limits if provided
        if xlims is not None:
            ax.set_xlim(xlims[0], xlims[1])
        if ylims is not None:
            ax.set_ylim(ylims[0], ylims[1])

        # Invert the y-axis for depth plots
        ax.invert_yaxis()

        # Set axis labels and grid
        ax.set_xlabel("Vp (km/s)")
        ax.set_ylabel(depth)
        ax.legend()
        ax.grid()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    return

def assign_global_colors(formations):
    """
    Assign consistent and distinguishable colors to formations globally.

    Args:
        formations (pd.Series): A pandas Series containing formation names.

    Returns:
        dict: A dictionary mapping each unique formation name to a distinct color.
    """
    # Get sorted unique formations to ensure consistent color assignment
    unique_formations = sorted(formations.unique())

    # Generate a color map with a sufficient number of distinct colors
    color_map = plt.cm.get_cmap("tab20", len(unique_formations))
    # color_map = plt.colormaps.get_cmap("tab20", len(unique_formations))

    # Create a dictionary mapping formations to their respective colors
    return {formation: color_map(i) for i, formation in enumerate(unique_formations)}
        
def plot_velocity_logs(
    data,
    formations=None,
    wells=None,
    stations=None,
    depth="Depth[km]",
    vel="Vp[km/s]",
    smooth_interval=None,
    xlims=None,
    ylims=None,
    region=None,
    scale_bar=None,
    grid=True,
    formations_key_order=None,
    axes_limit=6,
    minor_ticks=True,
    show=True,
    savefig=None,
    ):
    """
    Plot velocity logs for wells with optional formation boundaries and smoothing.

    Args:
        data (pd.DataFrame): DataFrame containing well velocity and depth data.
        formations (pd.DataFrame): DataFrame containing formation boundary data.
        wells (list): List of well names to include in the plot. If None, all wells are included.
        depth (str): Column name for depth data. Default is "Depth[km]".
        vel (str): Column name for velocity data. Default is "Vp[km/s]".
        smooth_interval (float): Interval for smoothing velocity data. Default is None.
        xlims (tuple): Limits for the x-axis (velocity). Default is None.
        ylims (tuple): Limits for the y-axis (depth). Default is None.
        region (tuple): Longitude/latitude limits for the map plot. Default is None.
        grid (bool): Whether to include gridlines. Default is True.
        formations_key_order (str): Key for ordering formations. Default is None.
        axes_limit (int): Maximum number of wells per plot. Default is 6.
        minor_ticks (bool): Whether to include minor gridlines. Default is True.
        show (bool): Whether to display the plot. Default is True.
        savefig (str): Path to save the figure. Default is None.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Filter the wells to be plotted. If no wells are provided, include all wells.
    sortby = wells
    if wells is None:
        wells = list(set(data["well_name"].tolist()))
    data = data[data["well_name"].isin(wells)]

    # Sort wells based on input order or by longitude.
    if sortby is None:
        data = data.sort_values(["longitude"])
        order = data["well_name"].unique()
    else:
        data["well_name"] = pd.Categorical(data["well_name"], categories=wells, ordered=True)
        data = data.sort_values("well_name")
        order = data["well_name"].unique()

    # Load reference velocity models (e.g., DB1D, Sheng (2022)).
    db1d_velmodel, sheng_velmodel = get_dw_models()

    # Define line styles for velocity models.
    vel_model_style = {
        "DB1D": {
            "linewidth": 1.5,
            "color": "black",
            "linestyle": "-",
            "vel_model": db1d_velmodel,
        },
        "Sheng (2022)": {
            "linewidth": 1.5,
            "color": "black",
            "linestyle": "--",
            "vel_model": sheng_velmodel,
        },
    }

    # Create legend elements for velocity models.
    vel_legend_handles = []
    for key, style in vel_model_style.items():
        legend_line = plt.Line2D(
            [0], [0], color=style["color"], lw=style["linewidth"], linestyle=style["linestyle"], label=key
        )
        vel_legend_handles.append(legend_line)

    # Assign global colors for formations if formation data is provided.
    if formations is not None:
        global_formation_colors = assign_global_colors(formations["FormationName"].drop_duplicates())
        if formations_key_order is None:
            formations_key_order = "FormationName"
        formations_order = formations.copy().drop_duplicates("FormationName")
        formations_order = formations_order.sort_values(formations_key_order, ignore_index=True)
        formations_order = formations_order["FormationName"].tolist()

    # Group data by wells.
    grouped_data = data.groupby("well_name")

    # Split wells into batches if there are more wells than axes_limit.
    n_wells = len(wells)
    if n_wells > axes_limit:
        real_order = [order[i:i + axes_limit] for i in range(0, len(order), axes_limit)]
    else:
        real_order = [order]

    # Plot each batch of wells.
    for k, batch_order in enumerate(real_order):
        # Map plot to show well and station locations.
        map_fig, map_ax = plt.subplots(figsize=(14, 8))
        wells_in_batch = [grouped_data.get_group(w).iloc[0] for w in batch_order]
        wells_in_batch = pd.DataFrame(wells_in_batch)

        # Plot stations if provided.
        if stations is not None:
            map_ax.scatter(
                stations["longitude"], stations["latitude"],
                marker="^", linestyle="None", color="gray", s=30, label="Stations"
            )

        # Plot wells on the map.
        for well_name, lon, lat in zip(wells_in_batch["well_name"], wells_in_batch["longitude"], wells_in_batch["latitude"]):
            map_ax.scatter(lon, lat, marker="*", s=200, edgecolor="black", linewidth=1, label="Log: " + well_name)

        # Set map region if provided.
        if region is not None:
            map_ax.set_xlim(region[0], region[1])
            map_ax.set_ylim(region[2], region[3])

        # Add gridlines.
        if grid:
            map_ax.grid(color="black", linewidth=0.5, linestyle=":")

        if scale_bar is not None:
            distance = scale_bar
            # Add scale bar.
            scalebar = AnchoredSizeBar(
                map_ax.transData,
                distance/100,
                f"{int(distance/2)} km",
                loc="lower right",
                pad=0.5,
                color="black",
                frameon=False,
                size_vertical=0.005,
            )
            scalebar2 = AnchoredSizeBar(
                map_ax.transData,
                distance/100/2,
                loc="lower right",
                label="",
                pad=0.5,
                color="white",
                frameon=False,
                size_vertical=0.005,
            )
            map_ax.add_artist(scalebar)
            map_ax.add_artist(scalebar2)

        # Add axis labels and legend.
        map_ax.set_xlabel("Longitude")
        map_ax.set_ylabel("Latitude")
        map_fig.legend(loc="upper left", ncol=1, bbox_to_anchor=(0.8, 0.7))
        map_fig.tight_layout(rect=[0, 0, 0.82, 1])

        # Save the map figure if savefig is provided.
        if savefig:
            base, ext = os.path.splitext(savefig)
            path_name = f"{base}_m{k}{ext}"
            map_fig.savefig(path_name,dpi=300)

        # Plot velocity logs.
        fig, axes = plt.subplots(1, len(batch_order), sharey=True, figsize=(12, 14))
        
        ## in case there is only one axis, it's necessary to put in a list
        if len(batch_order) == 1:
            axes = [axes]
        
        global_legend_handles = []

        # Process each well in the batch.
        for i, well_name in enumerate(batch_order):
            print("Well:", well_name)
            
            # Extract data for the current well and sort by depth.
            single_data = grouped_data.get_group(well_name).sort_values("TVD[km]")
            elevation_km = -round(single_data["elevation[km]"].iloc[0], 1)
            
            # Plot a horizontal line to represent the well's elevation.
            axes[i].axhline(y=elevation_km, color='black', linestyle='-', linewidth=1)
            
            # Add velocity model lines to the plot.
            for key, style in vel_model_style.items():
                vel_model = style["vel_model"]
                data2plot = vel_model.data.copy()
                data2plot["Depth (km)"] += elevation_km
                data2plot = data2plot[data2plot["Depth (km)"] >= elevation_km]
                axes[i].step(data2plot["VP (km/s)"], 
                            data2plot["Depth (km)"], 
                            color=style["color"], 
                            linewidth=style["linewidth"], 
                            linestyle=style["linestyle"], 
                            label=key)
            
            # Adjust depth based on elevation and apply smoothing if required.
            if depth == "Depth[km]":
                single_data[depth] = single_data["TVD[km]"] + elevation_km
            if smooth_interval is not None:
                single_data['Depth_interval'] = (single_data['Depth[km]'] // smooth_interval) * smooth_interval
                result = single_data.groupby('Depth_interval')['Vp[km/s]'].median().reset_index()
                result.columns = ['Depth[km]', 'Vp[km/s]']
                if result.empty:
                    print(f"{well_name} empty using smooth_interval = {smooth_interval} km")
                    continue
                else:
                    single_data = result
            
            # Plot the velocity log for the well.
            axes[i].step(single_data[vel], 
                        single_data[depth], 
                        "red", 
                        linewidth=2.5, 
                        label=well_name)

            # Plot formation boundaries for the well.
            if formations is not None:
                single_formation = formations[formations["API_UWI_12"] == well_name[:-3]]
                single_formation = single_formation.sort_values("TVD_Top", ignore_index=True)
                
                for _, f in single_formation.iterrows():
                    depth_f = (f["TVD_Top"] * 0.0003048) + elevation_km
                    formation_name = f["FormationName"]
                    color = global_formation_colors.get(formation_name, "gray")
                    axes[i].axhline(y=depth_f, color=color, linestyle="--", linewidth=1.5)
                    
                    # Add the formation to the legend if not already present.
                    if not any(handle.get_label() == formation_name for handle in global_legend_handles):
                        global_legend_handles.append(
                            plt.Line2D([0], [0], color=color, lw=2, linestyle="--", label=formation_name)
                        )
                        
            # Apply axis limits and invert y-axis for depth.
            if xlims:
                axes[i].set_xlim(xlims)
            if ylims:
                axes[i].set_ylim(ylims)
            axes[i].invert_yaxis()
            
            # Set labels and titles for the subplot.
            axes[i].set_title(well_name)
            axes[i].set_xlabel("Velocity [km/s]")
            
            # Add gridlines to the plot.
            if grid:
                axes[i].grid(color='black', linewidth=0.5, linestyle=":")
                if minor_ticks:
                    axes[i].minorticks_on()  # Enable minor ticks
                    axes[i].grid(color='gray', linewidth=0.5, linestyle=":", which='minor')
        
        # Sort and add legend for formations.
        if formations is not None:
            global_legend_handles = sorted(
                global_legend_handles,
                key=lambda handle: formations_order.index(handle.get_label()) if handle.get_label() in formations_order else float('inf')
            )
            fig.legend(handles=global_legend_handles, loc="upper left", fontsize=10, ncol=4, bbox_to_anchor=(0.05, 0.2))
        
        # Add velocity model legend to the figure.
        fig.legend(handles=vel_legend_handles, loc="upper left", fontsize=10, ncol=1, bbox_to_anchor=(0.8, 0.2))
        
        # Finalize layout and save or display the plot.
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.26)
        
        if savefig:
            base, ext = os.path.splitext(savefig)
            path_name = f"{base}_p{k}{ext}"
            fig.savefig(path_name,dpi=300)
        if show:
            plt.show()
        
        
    
if __name__ == "__main__":
    
    # 1 
    # folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/data"
    # well_pad_names = get_well_pad_names(folder)

    # 2
    # well_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/env_csv-Wells-6e346_2024-12-23.csv"
    # formation_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/env_csv-FormationTops-332ba_2024-12-23.csv"
    # formation = pd.read_csv(formation_path)
    # well = pd.read_csv(well_path)
    # folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/data"
    # data = get_sonics_data(folder,formation,well)
    # output = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/wells_aoi.csv"
    # data.to_csv(output,index=False)


    # 3
    # output = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/wells_aoi.csv"
    # data = pd.read_csv(output)
    # formations = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/env_csv-FormationTops-332ba_2024-12-23.csv"
    # formations = pd.read_csv(formations)
    # plot_velocity_logs(data,depth="Depth[km]",
    #                    ylims=(-2,6),
    #                    xlims=(1.5,6.5),
    #                    smooth_interval=0.1,
    #                    formations=formations)
    # plot_velocity_logs(data,depth="TVD[km]",ylims=(-2,6))


    ## enverus sheng
    
    ## eneverus get data
    # user = "emmanuel.castillotaborda@utdallas.edu"
    # pss = "Sismologia#1804"
    # download_folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOISheng/all_data"

    # df = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOISheng/env_csv-Logs-6e26d_2024-12-31.csv")

    # client = Client(user,pss)
    # client.query(log_df=df,
    #             download_folder=download_folder)

    # 2
    # well_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOISheng/env_csv-Wells-7d7bb_2024-12-31.csv"
    # formation_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOISheng/env_csv-FormationTops-e90fb_2024-12-31.csv"
    # formation = pd.read_csv(formation_path)
    # well = pd.read_csv(well_path)
    # folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOISheng/all_data"
    # data = get_sonics_data(folder,formation,well)
    # output = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOISheng/wells_aoiSheng_all_data.csv"
    # data.to_csv(output,index=False)

    output = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOISheng/wells_aoiSheng_all_data.csv"
    data = pd.read_csv(output)
    formations = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOISheng/env_csv-FormationTops-e90fb_2024-12-31.csv"
    formations = pd.read_csv(formations)
    
    stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/delaware_onlystations_160824.csv"
    stations = pd.read_csv(stations_path)
    region = [-103.3612838603971795,-103.1225914879300092,
              31.0731668393013969,31.2365797422041531]
    
    savefig = "/home/emmanuel/ecastillo/dev/delaware/10102024/figures/enverus_sheng/enverus_sheng.png"
    # plot_velocity_logs(data,depth="Depth[km]",
    #                    ylims=(-2,6),
    #                    xlims=(1.5,6.5),
    #                    smooth_interval=0.15,
    #                    scale_bar=10,
    #                    region=region,
    #                    stations=stations,
    #                    formations=formations,
    #                 #    savefig=savefig,
    #                    show=True
    #                 )


    ## eneverus get data
    # user = "emmanuel.castillotaborda@utdallas.edu"
    # pss = "Sismologia#1804"
    # download_folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/test"

    # df = pd.read_csv("/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/env_csv-Logs-158d7_2024-12-23.csv")

    # client = Client(user,pss)
    # client.query(log_df=df,
    #             download_folder=download_folder)
    
    ## all data
    
    # well_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/env_csv-Wells-6e346_2024-12-23.csv"
    # formation_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/env_csv-FormationTops-332ba_2024-12-23.csv"
    # formation = pd.read_csv(formation_path)
    # well = pd.read_csv(well_path)
    # folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/all_data"
    # data = get_sonics_data(folder,formation,well)
    # output = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/wells_aoi_all.csv"
    # data.to_csv(output,index=False)
    
    # output = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/wells_aoi_all.csv"
    # data = pd.read_csv(output)
    # formations = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/env_csv-FormationTops-332ba_2024-12-23.csv"
    # formations = pd.read_csv(formations)
    # stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/stations/delaware_onlystations_160824.csv"
    # stations = pd.read_csv(stations_path)
    # region = [-104.843290,-103.799420,
    #           31.396100,31.915050]
    
    # savefig = "/home/emmanuel/ecastillo/dev/delaware/10102024/figures/enverus/enverus_aoi.png"
    # plot_velocity_logs(data,depth="Depth[km]",
    #                    ylims=(-2,6),
    #                    xlims=(1.5,6.5),
    #                    smooth_interval=0.1,
    #                     scale_bar=50,
    #                    region=region,
    #                    stations=stations,
    #                    formations=formations,
    #                 #    wells = ["42-109-31362-00-00"],
    #                 #    savefig=savefig,
    #                    show=True
    #                 )