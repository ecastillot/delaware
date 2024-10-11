



if __name__ == "__main__":
    import os
    from delaware.utils import *
    
    x = (-104.84329,-103.79942)
    y = (31.3961,31.91505)
    z = (-2,10)
    nx, ny, nz = 30,40,60
    proj = "EPSG:3857"
    vel_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/vel/DB_model.csv"
    stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/stations/delaware_onlystations_160824.csv"
    
    
    vel_models = get_db1d_syn_vel_model(x,y,z,nx,ny,nz,vel_path,proj)
    stations = get_db_syn_stations(x,y,stations_path,proj)
    new_stations_path = os.path.join(os.path.dirname(stations_path),"standard_stations.csv")
    stations.data.to_csv(new_stations_path,index=False)
    
    # print(stations.data)
    # # p_model.plot_profile()
    vel_models["P"].plot_velocity_model(coords="geo",
                                        stations=stations,
                                        view_init_args={"elev":0, "azim":-90}
                                        )
    
    
    