from delaware.core.database import load_dataframe_from_sqlite
from delaware.core.read import EQPicks
import matplotlib.pyplot as plt
import seaborn as sns
# Enable LaTeX-style text rendering
plt.rc('text', usetex=False)

def create_joint_plot(db_path, author, 
                      ev_ids=None,
                      output_path=None):
    # Load DataFrame from SQLite database
    df = load_dataframe_from_sqlite(db_name=db_path)
    
    if ev_ids is not None:
        df = df[df["ev_id"].isin(ev_ids)]
    
    # Create a joint plot
    g = sns.jointplot(
        x="delta_arrival_time",
        y=f"sr_r [km]_{author}",
        hue="phase_hint",
        palette=["gray", "tomato"],
        marker="+",
        xlim=(-10, 10),
        ratio=2,
        data=df,
    )
    
    # Rename the axes
    g.set_axis_labels(r'$t_{\mathrm{manual}} - t_{\mathrm{synthetic}}$ (s)',
                      r'Source-Receiver Distance (km)',
                      fontsize=14)

    # Rename the legend title
    g.ax_joint.legend(title='Phase', fontsize=12, title_fontsize=14)

    if output is not None:
        # Save the figure with high DPI
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    return g

root = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi"
path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/picks/versus/versus_usgs_20170101_20240922VSpykonal_growclust.db"
author = "usgs_20170101_20240922"
author2 = "pykonal_growclust"
proj = "EPSG:3857"

eqpicks = EQPicks(root=root,
                  author=author2,
                  xy_epsg=proj,
                  catalog_header_line=0)

# # Example usage - all
# output = "/home/emmanuel/ecastillo/dev/delaware/10102024/figures/all_growclust_vs_synthetic.png"
# create_joint_plot(path, author, output)

# # # Example usage - r1
# output = "/home/emmanuel/ecastillo/dev/delaware/10102024/figures/r1_growclust_vs_synthetic.png"
# eqpicks.query(region_lims=[-104.6,-104.4,31.4,31.8])
# ev_ids = eqpicks.catalog.data["ev_id"].to_list()
# create_joint_plot(path, author,ev_ids=ev_ids,output_path=output)

# # Example usage - r2
# output = "/home/emmanuel/ecastillo/dev/delaware/10102024/figures/r2_growclust_vs_synthetic.png"
# eqpicks.query(region_lims=[-104.4,-104.2,31.4,31.8])
# ev_ids = eqpicks.catalog.data["ev_id"].to_list()
# create_joint_plot(path, author,ev_ids=ev_ids,output_path=output)

# # Example usage - r3
# output = "/home/emmanuel/ecastillo/dev/delaware/10102024/figures/r3_growclust_vs_synthetic.png"
# eqpicks.query(region_lims=[-104.2,-104.0,31.4,31.8])
# ev_ids = eqpicks.catalog.data["ev_id"].to_list()
# create_joint_plot(path, author,ev_ids=ev_ids,output_path=output)

# Example usage - r4
output = "/home/emmanuel/ecastillo/dev/delaware/10102024/figures/r4_growclust_vs_synthetic.png"
eqpicks.query(region_lims=[-104.0,-103.8,31.4,31.8])
ev_ids = eqpicks.catalog.data["ev_id"].to_list()
create_joint_plot(path, author,ev_ids=ev_ids,output_path=output)