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

fig,(ax0,ax1) = plt.subplots(1,2, 
                            #  figsize=(8, 12),
                             sharey=True,
                             gridspec_kw={'wspace': 0.05})

ax0.step(db1d_velmodel.data["VP (km/s)"], 
        db1d_velmodel.data["Depth (km)"], 'blue', 
        linewidth=2.5, linestyle='-', 
        label='DB1D')
ax0.step(sheng_velmodel.data["VP (km/s)"], 
        sheng_velmodel.data["Depth (km)"], 
        'orange', linewidth=2.5, linestyle='-', 
        label='Sheng (2022)')

ax0.legend(loc='lower left')

ax0.set_xlabel(r'$V_{p}$ [km/s]', fontsize=12)
ax0.set_ylabel('Depth [km]', fontsize=12)
ax0.set_ylim(ymin=0, ymax=10)
ax0.set_yticks(np.arange(0, 11, 1))  # Y-axis ticks every 1 unit
ax0.set_xlim(xmin=2, xmax=6.5)
ax0.invert_yaxis()
ax0.grid(True)

ax1.plot(v_mean_db1d, z_values, 'blue', 
        linewidth=2.5, linestyle='-', 
        label='DB1D')
ax1.plot(v_mean_sheng, z_values,'orange', linewidth=2.5, linestyle='-', 
        label='Sheng (2022)')
ax1.legend(loc='lower left')
ax1.set_xlabel(r'Average $V_{p}$ [km/s]', fontsize=12)
ax1.set_ylabel('')  # Set the label to an empty string
ax1.set_ylim(ymin=0, ymax=10)
ax1.set_xlim(xmin=2, xmax=6.5)
ax1.invert_yaxis()
ax1.grid()

output_path = os.path.join(dw_path,"script/vp/vp.png")
fig.savefig(output_path, dpi=300, bbox_inches='tight')

plt.show()

# z_values = np.linspace(0,10,100)
# v_mean_sheng = [sheng_velmodel.get_average_velocity(phase_hint="P", zmax=z) for z in z_values]
# v_mean_sheng = np.array(v_mean_sheng)