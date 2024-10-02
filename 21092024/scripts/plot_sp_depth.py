from delaware.loc.s_p import plot_montecarlo_depths_by_station,plot_montecarlo_depths
import pandas as pd
import numpy as np

#specific model

# output = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/growclust/depth_simple_vel_6km.db"
output = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/growclust/depth_simple_specific_vel.db"
output_fig = "/home/emmanuel/ecastillo/dev/delaware/21092024/output/figures/sp/depth_simple_specific_vel.png"
# plot_montecarlo_depths(output)
plot_montecarlo_depths_by_station(output,savefig=output_fig)


# general model
# # output = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/growclust/depth_simple_vel_6km.db"
# output = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/growclust/depth_simple_general_vel.db"
# output_fig = "/home/emmanuel/ecastillo/dev/delaware/21092024/output/figures/sp/depth_simple_general_vel.png"
# # plot_montecarlo_depths(output)
# plot_montecarlo_depths_by_station(output,savefig=output_fig)