



if __name__ == "__main__":
    import os
    from delaware.utils import *
    from delaware.synthetic.tt import EarthquakeTravelTimeDataset
    
    x = (-104.84329,-103.79942)
    y = (31.3961,31.91505)
    z = (-2,10)
    nx, ny, nz = 50,40,60
    proj = "EPSG:3857"
    vel_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/vel/DB_model.csv"
    stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/stations/delaware_onlystations_160824.csv"
    tt_folder_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/synthetics/tt"
    
    stations = get_db_syn_stations(x,y,stations_path,proj)
    x,y,z,profiles = prepare_db1d_syn_vel_model(x,y,z,vel_path,proj)
    # print("aca",x,y,z)
    # exit()
    print(stations)
    
    
    # print(x,y,z,profiles)
    # exit()
    for phase, vel in profiles.items():
        print(phase)
        tt_path = os.path.join(tt_folder_path,f"{phase}_tt.h5")
        tt = EarthquakeTravelTimeDataset(phase,stations)
        tt.add_grid_with_velocity_model(x,y,z,nx,ny,nz,
                        xy_epsg=proj,
                        vel1d=vel)
        tt.save_traveltimes(output=tt_path)
    
    
    
    