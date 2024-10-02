from delaware.vel.velocity import ScalarVelModel,ScalarVelPerturbationModel
import pandas as pd
import numpy as np

output = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/simple_vel.npz"

svm = ScalarVelModel(4.5,2.4,"simple_sp_vel")
svm.get_perturbation_model(output=output)

svpm = ScalarVelPerturbationModel(output)
svpm.plot()
