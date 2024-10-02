from delaware.loc.s_p import SP_Database
from delaware.vel.velocity import ScalarVelModel,ScalarVelPerturbationModel
import pandas as pd
import numpy as np


# catalog_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/nlloc/catalog_sp_method.db"
# picks_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/nlloc/picks_sp_method.db"


# # specific vel model
output = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/vel/simple_specific_vel.npz"
svpm = ScalarVelPerturbationModel(output)
catalog_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/growclust/catalog_sp_method.db"
picks_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/growclust/picks_sp_method.db"


sp_db = SP_Database(catalog_path=catalog_path,
                    picks_path=picks_path)


output = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/growclust/depth_simple_specific_vel.db"
sp_db.run_montecarlo(svpm,output=output)

# # general vel model
# output = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/vel/simple_general_vel.npz"
# svpm = ScalarVelPerturbationModel(output)
# catalog_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/growclust/catalog_sp_method.db"
# picks_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/growclust/picks_sp_method.db"


# sp_db = SP_Database(catalog_path=catalog_path,
#                     picks_path=picks_path)


# output = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/growclust/depth_simple_general_vel.db"
# sp_db.run_montecarlo(svpm,output=output)