from delaware.vel.velocity import VelModel,MontecarloVelModel
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
out = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/vel/montecarlo.db"
# vel_model.monte_carlo_simulation(out,num_simulations=1000, 
#                                  mean=0, std_dev=0.3)

mv = MontecarloVelModel(out)
mv.plot(vel_model,n_bins=50)