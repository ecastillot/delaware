from delaware.vel.velocity import ScalarVelModel,ScalarVelPerturbationModel
import pandas as pd
import numpy as np


output = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/vel/simple_general_vel.npz"
output_fig = "/home/emmanuel/ecastillo/dev/delaware/21092024/output/velocity/simple_general_vel.png"

min_p_vel, max_p_vel = 4.405,6.061
min_s_vel, max_s_vel = 2.517,3.463

vp_disp = round((max_p_vel -min_p_vel)/2,2)
vs_disp = round((max_s_vel -min_s_vel)/2,2)

vp = min_p_vel+vp_disp
vs = min_s_vel+vs_disp 



svm = ScalarVelModel(vp,vs,"simple_gen_vel")
svm.get_perturbation_model(output=output,
                           p_std_dev=vp_disp,
                           s_std_dev=vs_disp)

svpm = ScalarVelPerturbationModel(output)
svpm.plot(savefig=output_fig)

# output = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/vel/simple_vel_6km.npz"

# svm = ScalarVelModel(5.917,3.381,"simple_vel_6km")
# svm.get_perturbation_model(output=output,p_std_dev=0.5,
#                            s_std_dev=0.5)

# svpm = ScalarVelPerturbationModel(output)
# svpm.plot()
