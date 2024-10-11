import pandas as pd
from GPA_01092024.tt_utils  import Stations,get_xyz_velocity_model,single_latlon2xy_in_km

stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/stations/delaware_onlystations_160824.csv"
proj = "EPSG:3857"

x = (-104.84329,-103.79942)
y = (31.3961,31.91505)
z = (-2,10)
nx, ny, nz = 30,40,60

# see the velocity here:
# https://pubs.geoscienceworld.org/ssa/tsr/article/2/1/29/611100/On-the-Depth-of-Earthquakes-in-the-Delaware-Basin
p_vel = {"depth":[-2,0.06,0.15,0.5,1,1.5,2,2.5,3.25,4.60,5.20,5.35,5.8,6.2,10],
        "vel":[2.1,3,4.3,4.55,6.10,4.3,4.5,3.9,3.65,3.80,4.75,5.8,5.3,6,6]}

# converting x y in degrees to km
xmin, ymin = single_latlon2xy_in_km(y[0],x[0],proj)
xmax, ymax= single_latlon2xy_in_km(y[1],x[1],proj)
x = (xmin,xmax)
y = (ymin,ymax)


# preparing stations data
stations_data = pd.read_csv(stations_path)
stations_data.rename(columns={
                            "station_elevation":"elevation"},
                            )
stations_data["station_index"] = stations_data.index
stations_data["elevation"] = stations_data["elevation"]/1e3

#stations inside the polygon
dw_w_pol = [(x[1],y[0]),
        (x[1],y[1]),
        (x[0],y[1]),
        (x[0],y[0]),
        (x[1],y[0])]
stations = Stations(stations_data,xy_epsg=proj)
stations.filter_region(polygon=dw_w_pol)
# print(stations.get_minmax_coords())
# print(stations.data["z[km]"].min())


# vel model
model = get_xyz_velocity_model(x,y,z,nx,ny,nz,phase="P",
                        xy_epsg=proj,
                        profile=p_vel,layer=True)
# model.plot_velocity_model(coords="npts")
# model.plot_velocity_model()
print(model.geo_min_coords)
print(model.geo_max_coords)
print(model.geo_coords)
# print(model.geo_min_coords)
# print(model.min_coords)
# print(model.max_coords)




