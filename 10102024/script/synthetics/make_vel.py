



if __name__ == "__main__":
    import os
    from delaware.utils import *
    from delaware.synthetic.utils import get_db1d_syn_vel_model
    
    x = (-104.84329,-103.79942)
    y = (31.3961,31.91505)
    z = (-2,20)
    nx, ny, nz = 100,60,100
    proj = "EPSG:3857"
    vel_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/vel/DB_model.csv"
    stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/stations/delaware_onlystations_160824.csv"
    
    
    vel_models = get_db1d_syn_vel_model(x,y,z,nx,ny,nz,vel_path,proj)
    stations = get_db_stations(stations_path,x,y,proj)
    new_stations_path = os.path.join(os.path.dirname(stations_path),"standard_stations.csv")
    stations.data.to_csv(new_stations_path,index=False)
    
    vel_models["P"].plot_velocity_model(coords="geo",
                                        stations=stations,
                                        view_init_args={"elev":0, "azim":-90}
                                        )
    
    
    