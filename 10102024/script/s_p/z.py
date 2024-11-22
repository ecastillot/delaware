from delaware.vel.vel import VelModel
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

version = "10102024"
dw_path = os.path.join("/home/emmanuel/ecastillo/dev/delaware",version)

vel_path = os.path.join(dw_path,"data/vel/DB_model.csv")
data = pd.read_csv(vel_path)
db1d_velmodel = VelModel(data=data,dtm=-0.7,name="db1d")

vel_path = os.path.join(dw_path,"data/vel/DW_model.csv")
data = pd.read_csv(vel_path)
sheng_velmodel = VelModel(data=data,dtm=-0.7,name="Sheng")

z_values = np.linspace(0,10,100)
v_mean_db1d = [db1d_velmodel.get_average_velocity(phase_hint="P", zmax=z) \
                                                    for z in z_values]
v_mean_db1d = np.array(v_mean_db1d)
v_mean_sheng = [sheng_velmodel.get_average_velocity(phase_hint="P", zmax=z)\
                                                        for z in z_values]
v_mean_sheng = np.array(v_mean_sheng)

vp = np.linspace(2,6,100)
# t_sp = 1
vps = 1.72
z1 = vp*(0.7/(vps-1))
z2 = vp*(0.9/(vps-1))

# print(z)
# exit()

fig,axes = plt.subplots(1,6, 
                            #  figsize=(8, 12),
                             sharey=True,
                             gridspec_kw={'wspace': 0.05})

for i in range(0,6):

        axes[i].plot(vp,z1,'k',linewidth=1.5)
        axes[i].plot(vp,z2,'k',linewidth=1.5)
        axes[i].fill_between(vp, z1, z2, color='gray', alpha=0.3)  # Color span between y1 and y2
        axes[i].plot(v_mean_db1d, z_values, 'blue', 
                linewidth=1.5, linestyle='-', 
                label='DB1D')
        axes[i].plot(v_mean_sheng, z_values,'orange', linewidth=1.5, linestyle='-', 
                label='Sheng (2022)')
        # axes[i].legend(loc='lower left')
        axes[i].set_xlabel(r'Average $V_{p}$ [km/s]', fontsize=12)
        axes[i].set_ylabel('')  # Set the label to an empty string
        axes[i].set_ylim(ymin=0, ymax=10)
        axes[i].set_xlim(xmin=2, xmax=6.5)
        axes[i].invert_yaxis()
        axes[i].grid()

# output_path = os.path.join(dw_path,"script/vp/vp.png")
# fig.savefig(output_path, dpi=300, bbox_inches='tight')

plt.show()

# z_values = np.linspace(0,10,100)
# v_mean_sheng = [sheng_velmodel.get_average_velocity(phase_hint="P", zmax=z) for z in z_values]
# v_mean_sheng = np.array(v_mean_sheng)