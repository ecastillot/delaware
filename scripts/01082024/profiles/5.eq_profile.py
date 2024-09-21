import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pygmt
from EQViewer.eqviewer import Profile, BaseProfile
from EQViewer.eqviewer import Catalog, BasePlot
from EQViewer.eqviewer import CPT, MulCatalog

def load_and_preprocess_data(catalogpath):
    """
    Load and preprocess the earthquake catalog data.

    Args:
        catalogpath (str): Path to the CSV file containing the earthquake data.

    Returns:
        pd.DataFrame: Processed DataFrame with renamed columns.
    """
    df = pd.read_csv(catalogpath, parse_dates=["origin_time"])
    df = df.rename(columns={"latR": "latitude",
                            "lonR": "longitude",
                            "depR": "depth",
                            "mag": "magnitude"})
    return df

def create_catalog(df):
    """
    Create a Catalog object with the provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing earthquake data.

    Returns:
        Catalog: Catalog object initialized with the provided data.
    """
    cat_baseplot = BasePlot(color="gray",
                            style="c0.1c",
                            size=None,
                            cmap=False,
                            pen="black",
                            label="all_events")
    catalog = Catalog(data=df, baseplot=cat_baseplot)
    return catalog

def filter_catalog_region(catalog, filter_domain):
    """
    Filter the catalog to a specific region defined by a polygon.

    Args:
        catalog (Catalog): The Catalog object to be filtered.
        filter_domain (list): Coordinates defining the polygon for filtering.
        
    Returns:
        Catalog: Filtered Catalog object.
    """
    pol2filter = [(filter_domain[0], filter_domain[2]),
                  (filter_domain[0], filter_domain[3]),
                  (filter_domain[1], filter_domain[3]),
                  (filter_domain[1], filter_domain[2]),
                  (filter_domain[0], filter_domain[2])]
    catalog = catalog.copy()
    catalog.filter_region(polygon=pol2filter)
    return catalog

def create_cpt(catalog):
    """
    Create a CPT object for color mapping.

    Args:
        catalog (Catalog): The Catalog object used to determine depth range.

    Returns:
        CPT: CPT object for color mapping.
    """
    cat_cpt = CPT(color_target="depth",
                  label="Depth (km)",
                  cmap="hot",
                  series=[catalog.data.depth.min(), catalog.data.depth.max()],
                  reverse=True,
                  overrule_bg=True)
    return cat_cpt

def plot_catalog(catalog, cpt):
    """
    Plot the catalog data on a map.

    Args:
        catalog (Catalog): The Catalog object to plot.
        cpt (CPT): CPT object for color mapping.

    Returns:
        pygmt.Figure: Figure object for further manipulation.
    """
    mulcatalog = MulCatalog([catalog], cpt=cpt, show_cpt=True)
    return mulcatalog.plot_map()

def create_profile():
    """
    Create a BaseProfile and Profile for plotting.

    Returns:
        Profile: Profile object initialized with the base profile.
    """
    baseprofile = BaseProfile(projection="x15c/-10c", depth_lims=[0, 10],
                              grid=(1, 1),
                              output_unit="km")
    profile = Profile(name=("A", "A'"),
                      coords=((-104.84329, 31.65), (-103.79942, 31.65)),
                      width=(-20, 20),
                      baseprofile=baseprofile)
    return profile

def add_catalog_to_profile(profile, mulcatalog):
    """
    Add a MulCatalog object to the Profile.

    Args:
        profile (Profile): The Profile object to which the catalog will be added.
        mulcatalog (MulCatalog): The MulCatalog object to be added.
    """
    profile.add_mulobject(mulcatalog, depth_unit="km")

def plot_lon_depth(catalog):
    """
    Plot lon vs depth with discrete color scaling for magnitude.

    Returns:
        tuple: Figure and Axes objects.
    """
    proj = catalog.data
    
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('tab20', 5)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(0, 6, 1), ncolors=5)
    scatter = ax.scatter(proj['longitude'], proj['depth'],
                         c=np.clip(proj['magnitude'], 0, 5),
                         cmap=cmap, alpha=0.6, edgecolors="w", linewidth=0.5, norm=norm)
    ax.invert_yaxis()
    # ax.set_xlabel('LOngitude ()', fontsize=14)
    ax.set_ylabel('Depth (km)', fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(0, 6, 1))
    cbar.set_label('Magnitude', fontsize=14)
    ax.set_ylim(10, 0)
    ax.set_xlim(-104.84329, -102.9081)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect(0.1)
    plt.tight_layout()
    return fig, ax

def main(df, region,filter_dates=None,output=None):
    """
    Main function to process and plot earthquake catalog data.

    Args:
        df (pd.DataFrame): DataFrame containing earthquake data.
        region (list): Coordinates defining the region to filter.
    """
    # Create and filter catalog
    catalog = create_catalog(df)
    catalog = filter_catalog_region(catalog, region)
    if filter_dates is not None:
        catalog.filter("origin_time",start=filter_dates[0],end=filter_dates[1])
    
    print(catalog.__str__(True))
    
    # Create CPT and plot catalog
    cpt = create_cpt(catalog)
    # mulcatfig = plot_catalog(catalog, cpt)
    
    # Create and update profile
    # profile = create_profile()
    # add_catalog_to_profile(profile, MulCatalog([catalog], cpt=cpt, show_cpt=True))
    
    # # Plot profile
    # profilefig = profile.plot_in_map(fig=mulcatfig, rescale=False)
    # profilefig.show()
    
    # Add catalog again to ensure correct rendering
    # add_catalog_to_profile(profile, MulCatalog([catalog], cpt=cpt, show_cpt=True))
    
    # Extract projection data and plot distance vs depth
    # proj = profile.mulobjects["MulCatalog"]["projections"][0]
    fig, ax = plot_lon_depth(catalog)
    
    if output is not None:
        # Save the figure if needed
        fig.savefig(output, dpi=300)

if __name__ == "__main__":
    import datetime as dt
    
    # File path to the earthquake catalog
    catalogpath = "/mnt/Ecopetrol/Analytics/emmanuel/dev/others/texnet_hirescatalog_fixed.csv"
    
    # Load and preprocess data
    df = load_and_preprocess_data(catalogpath)
    
    # Define the region for filtering
    filter_domain = [-104.84329, -102.9081, 30.7996, 31.91505]
    
    # filter_dates = (dt.datetime(2000,3,1),dt.datetime(2020,2,29) ) #x1
    filter_dates = (dt.datetime(2020,3,1),dt.datetime(2021,11,30)) #x2
    # filter_dates = (dt.datetime(2021,12,1),dt.datetime(2022,4,30)) #x3
    # filter_dates = (dt.datetime(2022,5,1),dt.datetime(2024,2,29)) #x4
    # filter_dates = (dt.datetime(2024,3,1),dt.datetime(2025,1,1)) #x5
    output= "/home/emmanuel/ecastillo/dev/delaware/profiles_delaware/x2.png"
    
    # Call main function with DataFrame and region
    main(df, filter_domain,filter_dates,output)
    plt.show()
