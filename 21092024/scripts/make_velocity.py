from delaware.vel.velocity import VelModel,VelPerturbationModel
import pandas as pd


data = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/vel/DB_model.csv"
data = pd.read_csv(data)

vel_model = VelModel(data,name="DB1D",dtm=-0.7)
# vel_model.plot_profile()


# # Add Gaussian noise to VP
# print(vel_model.data)
# data = vel_model.add_gaussian_perturbation('VP (km/s)', mean=0, std_dev=0.05)
# print(data)


# # Plot Monte Carlo Simulation
out = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/vel/montecarlo2.db"
vel_model.get_perturbation_model(out,n_perturbations=1000, 
                                 p_mean=0, p_std_dev=0.5,
                                 s_mean=0, s_std_dev=0.5)

mv = VelPerturbationModel(out)
mv.plot(vel_model,n_bins=50)