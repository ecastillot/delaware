import pandas as pd
import glob
import os
import lasio
import numpy as np
import matplotlib.pyplot as plt
from delaware.vel.vel import VelModel
from matplotlib import cm

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
    basename = os.path.basename(las_path)
    well_name = basename.split("_")[0]
    uwi12 = well_name[:-3]
    return uwi12
    

def get_sonic_data(las_path,formation):
    las = lasio.read(las_path)
    df = las.df()
    df = df.reset_index()
    # print(df.columns)
    
    uwi12 = get_uwi12(las_path)
    
    formation = formation[formation["API_UWI_12"] == uwi12]
    formation = formation.sort_values("MD_Top")
    
    md_values = None
    for x in ["DEPTH","DEPT","DEP"]:
        # print(df.columns.to_list())
        if x in df.columns.to_list():
            md_values = df[x]
            # print(x,df.columns.to_list())
            continue
            
    # print(md_values)
    if md_values is None:
        raise Exception("No Depth")
    
    dt_columns = []    
    if "DT" in df.columns.to_list():
        dt_columns.append("DT")
    if "DT_S" in df.columns.to_list():
        dt_columns.append("DT_S")
    if not dt_columns:
        raise Exception(f"No DT or DTS in {uwi12}")
    
    md_real_values = [md_values.min()] + formation["MD_Top"].to_list() + [md_values.max()]
    tvd_real_values = [md_values.min()] + formation["TVD_Top"].to_list() + [md_values.max()]
    tvd_values = np.interp(md_values, 
                           md_real_values, 
                           tvd_real_values)
    
    result = pd.DataFrame({"MD": md_values, "TVD": tvd_values})
    result = pd.concat([result,df[dt_columns]],axis=1)
    result.dropna(subset=dt_columns,inplace=True)

    # Convert MD and TVD from ft to km
    result["MD[km]"] = result["MD"] * 0.0003048
    result["TVD[km]"] = result["TVD"] * 0.0003048

    # result["Vp[km/s]"] = np.nan
    # result["vs[km/s]"] = np.nan
    # Convert DT (µs/ft) to Velocity (km/s)
    for dt in dt_columns:
        if dt == "DT":
            result["Vp[km/s]"] = 1 / (result["DT"] * 1e-6 * 3280.84)
        elif dt == "DT_S":
            result["Vs[km/s]"] = 1 / (result["DT"] * 1e-6 * 3280.84)
            
    result.dropna(subset=dt_columns,inplace=True)
    
    if result.empty:
        raise Exception("No Vp or Vs data")
    
    # print(result)
    # # Plot a specific log
    # plt.figure(figsize=(6, 10))
    # plt.plot(result['V[km/s]'], result["TVD[km]"], label="Vel", color='green')
    # plt.gca().invert_yaxis()
    # plt.xlabel("Vel [km/s]")
    # plt.ylabel("Depth (m)")
    # plt.title("Vel Log")
    # plt.legend()
    # plt.grid()
    # plt.show()
    # print(result)
    return result
    
def get_sonics_data(las_folder,formation,well):
    
    # Get all LAS file paths in the specified folder
    paths = glob.glob(os.path.join(las_folder, "*.LAS"))
    # print(get_well_names(las_folder))
    
    wells =  {}
    # p_well_names = []
    for las_path in paths:
        
        # Get the file name without extension
        basename = os.path.splitext(os.path.basename(las_path))[0]
        
        # Extract the well name (assumes it's the first part of the file name before an underscore)
        well_name = basename.split("_")[0]
        
               
        uwi12 = get_uwi12(las_path)
        single_well = well[well["API_UWI_12"] == uwi12]
        # print(single_well)
        # exit()
        single_well = single_well.iloc[0]
        latitude = single_well["Latitude"]
        longitude = single_well["Longitude"]
        elevation = single_well["ENVElevationGL_FT"]*0.0003048
        # print(lat,lon)
        
        # exit()
        
        
        # print(las_path)
        try:
            df = get_sonic_data(las_path,formation)
            # print(well_name,"OK",latitude,
            #       longitude,elevation)
        except Exception as e:
            continue
    
        df.insert(0,"well_name",well_name)
        df.insert(1,"latitude",latitude)
        df.insert(2,"longitude",longitude)
        df.insert(3,"elevation[km]",elevation)
    
        if well_name not in list(wells.keys()):
            print(well_name,"OK\t",latitude,
                  longitude,elevation)
        #     prev_max_depth = wells[well_name]["TVD[km]"].max()
        #     curr_max_depth = df["TVD[km]"].max()
        #     print(prev_max_depth,curr_max_depth)
        #     continue
    
        # p_well_names.append(well_name)
        wells[well_name] = df
        
    wells = pd.concat(list(wells.values()))
    # print(wells.describe())
    
    return wells

def get_dw_models():
    dw_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/vel"
    vel_path = os.path.join(dw_path,"DW_model.csv")
    data = pd.read_csv(vel_path)
    sheng_velmodel = VelModel(data=data,dtm=-0.7,name="Sheng")
    
    vel_path = os.path.join(dw_path,"DB_model.csv")
    data = pd.read_csv(vel_path)
    db1d_velmodel = VelModel(data=data,dtm=-0.7,name="db1d")
    return db1d_velmodel, sheng_velmodel

def plot_all_velocity_logs(data, 
                       formations=None,
                       wells=None, 
                       depth="Depth[km]", 
                       vel="Vp[km/s]", 
                       smooth="moving_average",
                       xlims= None,
                       ylims=None):
    grouped_data = data.groupby("well_name").__iter__()
    
    if wells is None:
        wells = list(set(data["well_name"].tolist()))
    
    if smooth is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))  # Adjusted figure size for better clarity
    
    db1d_velmodel, sheng_velmodel = get_dw_models()
    
    for well_name, single_data in grouped_data:
        single_data = single_data.sort_values("TVD[km]")
        
        if well_name not in wells:
            continue
        
        api12 = well_name[0:-3]
        # print(api12)
        single_formation = formations[formations["API_UWI_12"]==api12]
        single_formation = single_formation.drop_duplicates("FormationName",
                                                            ignore_index=True)
        # print(single_formation)
        
        if depth == "Depth[km]":
            single_data[depth] = single_data["TVD[km]"] - single_data["elevation[km]"]
        
        if smooth == "moving_average":
            single_data['Vp_smoothed'] = single_data['Vp[km/s]'].rolling(window=1000, 
                                                                         center=True).mean()
        
        for i in range(len(axes)):
            if i == 0:
                axes[i].step(single_data[vel], 
                        single_data[depth], 
                        label=well_name)
            elif i == 1:
                axes[i].step(single_data['Vp_smoothed'], 
                        single_data[depth], 
                        label=well_name)

    if not single_formation.empty:
        # print(single_formation)
        print(single_data["elevation[km]"].iloc[0])
        for row,f in single_formation.iterrows():
            depth_f = (f["TVD_Top"] * 0.0003048) - single_data["elevation[km]"].iloc[0]
            label_f = f["FormationName"]
            axes[i].axhline(y=depth_f, color='k', linestyle='--', linewidth=0.5)

            axes[i].text(1.02, depth_f, label_f, color='black', fontsize=8,
                    ha='left', va='top', 
                    transform=axes[i].get_yaxis_transform())  # Label
    # x_min, x_max = data['Vp_smoothed'].min(), data['Vp_smoothed'].max()
    # y_min, y_max = data[depth].min(), data[depth].max()
    
    # Set the same x and y limits for both subplots
    for ax in axes:
        ax.step(db1d_velmodel.data["VP (km/s)"], 
                db1d_velmodel.data["Depth (km)"], 'blue', 
                linewidth=2.5, linestyle='-', 
                label='DB1D')
        ax.step(sheng_velmodel.data["VP (km/s)"], 
                sheng_velmodel.data["Depth (km)"], 
                'orange', linewidth=2.5, linestyle='-', 
                label='Sheng (2022)')
        if xlims is not None:
            ax.set_xlim(xlims[0],xlims[1])
        if ylims is not None:
            ax.set_ylim(ylims[0],ylims[1])
        ax.invert_yaxis()  # Invert y-axis
        
        ax.set_xlabel("Vp (km/s)")
        ax.set_ylabel(depth)
        ax.legend()
        ax.grid()

    plt.tight_layout()  # Adjust layout for clarity
    plt.show()
    
    return

# Assign colors to formations globally
def assign_global_colors(formations):
    """Assign consistent colors to formations globally."""
    unique_formations = sorted(formations.unique())  # Sort to ensure consistency
    color_map = plt.cm.get_cmap("tab20", len(unique_formations))  # Use tab20 for distinct colors
    return {formation: color_map(i) for i, formation in enumerate(unique_formations)}
        
def plot_velocity_logs(data, 
                       formations=None,
                       wells=None, 
                       depth="Depth[km]", 
                       vel="Vp[km/s]", 
                       smooth="moving_average",
                       xlims= None,
                       ylims=None):
    
    sortby = wells
    if wells is None:
        wells = list(set(data["well_name"].tolist()))
    
    data = data[data["well_name"].isin(wells)]
    
    if sortby is None:
        data = data.sort_values(["longitude"])
        order = data["well_name"].unique()
    else:
        data["well_name"] = pd.Categorical(data["well_name"], categories=wells, ordered=True)
        data = data.sort_values("well_name")
        order = data["well_name"].unique()
        
    grouped_data = data.groupby("well_name")
    
    # if smooth is not None:
    fig, axes = plt.subplots(1, len(wells), 
                             figsize=(10, 10)# Adjusted figure size for better clarity
                                )  
    
    db1d_velmodel, sheng_velmodel = get_dw_models()
    
    for i,well_name in enumerate(order):
        single_data = grouped_data.get_group(well_name)
        single_data = single_data.sort_values("TVD[km]")
        
        
        # print(single_formation)
        
        #elevation
        elevation_km = - single_data["elevation[km]"].iloc[0]
        axes[i].axhline(y= elevation_km, 
                        color='black', linestyle='-', linewidth=1)
        
        axes[i].step(db1d_velmodel.data["VP (km/s)"], 
                    db1d_velmodel.data["Depth (km)"], 'blue', 
                    linewidth=2.5, linestyle='-', 
                    label='DB1D')
        axes[i].step(sheng_velmodel.data["VP (km/s)"], 
                sheng_velmodel.data["Depth (km)"], 
                'orange', linewidth=2.5, linestyle='-', 
                label='Sheng (2022)')
        
        if depth == "Depth[km]":
            single_data[depth] = single_data["TVD[km]"] + elevation_km
        
        if smooth == "moving_average":
            single_data['Vp_smoothed'] = single_data['Vp[km/s]'].rolling(window=100, 
                                                                         center=True).mean()
        
        
        if smooth is not None:
            axes[i].step(single_data['Vp_smoothed'], 
                        single_data[depth], 
                        "red",
                        label=well_name)
        else:
            axes[i].step(single_data[vel], 
                            single_data[depth], 
                            "red",
                            label=well_name)

        # if not single_formation.empty:
            # print(single_formation)
            # print(single_data["elevation[km]"].iloc[0])
            # for row,f in single_formation.iterrows():
            #     depth_f = (f["TVD_Top"] * 0.0003048) - single_data["elevation[km]"].iloc[0]
            #     label_f = f["FormationName"]
            #     axes[i].axhline(y=depth_f, color='k', linestyle='--', linewidth=0.5)

            #     axes[i].text(1.02, depth_f, label_f, color='black', fontsize=8,
            #             ha='left', va='top', 
            #             transform=axes[i].get_yaxis_transform())  # Label
        
        
        ##formations
        api12 = well_name[0:-3]
        # print(api12)
        single_formation = formations[formations["API_UWI_12"]==api12]
        single_formation = single_formation.sort_values("TVD_Top",
                                                            ignore_index=True)
        single_formation = single_formation.drop_duplicates("FormationName",
                                                            ignore_index=True)
            
        # Inside your main function
        # Assign global colors based on all formations
        all_formations = formations["FormationName"].drop_duplicates()
        global_formation_colors = assign_global_colors(all_formations)

        # Modify the plotting code
        if not single_formation.empty:
            legend_handles = []
            plotted_formations = set()  # Track formations already added to the legend for this axis
            
            for _, f in single_formation.iterrows():
                depth_f = (f["TVD_Top"] * 0.0003048) - single_data["elevation[km]"].iloc[0]
                formation_name = f["FormationName"]
                color = global_formation_colors.get(formation_name, "gray")
                
                # Plot hlines with consistent global colors
                axes[i].axhline(y=depth_f, color=color, linestyle="--", linewidth=1)
                
                # Build the legend dynamically based on the plotting order
                if formation_name not in plotted_formations:
                    legend_handles.append(
                        plt.Line2D([0], [0], color=color, lw=2, label=formation_name)
                    )
                    plotted_formations.add(formation_name)
            
            # Add legend for formations
            axes[i].legend(handles=legend_handles, loc="lower left", fontsize=8)
        
        
        if xlims is not None:
            axes[i].set_xlim(xlims[0],xlims[1])
        if ylims is not None:
            axes[i].set_ylim(ylims[0],ylims[1])
        axes[i].invert_yaxis()  # Invert y-axis
        
        axes[i].set_title(well_name)
        # axes[i].grid()
    

    plt.tight_layout()  # Adjust layout for clarity
    plt.show()
    
    return        
        
    
    
    
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
output = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/wells_aoi.csv"
data = pd.read_csv(output)
formations = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/env_csv-FormationTops-332ba_2024-12-23.csv"
formations = pd.read_csv(formations)
# plot_velocity_logs(data,depth="TVD[km]")
plot_velocity_logs(data,depth="Depth[km]",ylims=(-2,6),
                   formations=formations)
# plot_velocity_logs(data,depth="TVD[km]",ylims=(-2,6))
