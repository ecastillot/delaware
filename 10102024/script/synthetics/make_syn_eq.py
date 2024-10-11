



if __name__ == "__main__":
    import os
    from delaware.utils import *
    from delaware.synthetic.tt import EarthquakeTravelTime
    from delaware.synthetic.tt_utils import Earthquakes
    
    x = (-104.84329,-103.79942)
    y = (31.3961,31.91505)
    z = (-2,10)
    nx, ny, nz = 30,40,60
    proj = "EPSG:3857"
    stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/stations/standard_stations.csv"
    tt_folder_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/synthetics/tt"
    eq_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/growclust/texnet_hirescatalog.csv"
    syn_picks_folder_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/pykonal"
    
    df = pd.read_csv(stations_path)
    stations = Stations(data=df, xy_epsg=proj)
    print(stations)
    # growclust = get_texnet_high_resolution_catalog(eq_path)
    # eqs = Earthquakes(data=growclust.data,xy_epsg=proj)
    
    df = pd.DataFrame.from_dict({"latitude":[31.636749999999999],
                             "longitude":[-103.998790000000000],
                            #  "depth":[6.903000000000000],
                             "depth":[2.5000000000000],
                            #  "origin_time":[dt.datetime(2022,11,16,21,32,48.4819999)]
                             "origin_time":["2022-11-16 21:32:48.4819999"]
                             })
    df["origin_time"] = pd.to_datetime(df['origin_time'])
    print(df)
    earthquakes = Earthquakes(data=df, xy_epsg=proj)
    print(earthquakes)
    
    picks = []
    for phase in ("P","S"):
        tt_path = os.path.join(tt_folder_path,f"{phase}_tt.npz")
        ott_path = os.path.join(tt_folder_path,f"{phase}_tt.csv")
        eq = EarthquakeTravelTime(phase=phase, stations=stations,
                            earthquakes=earthquakes)
        print(tt_path)
        eq.load_velocity_model(path=tt_path,
                                    xy_epsg=proj)
        tt = eq.get_traveltimes(merge_stations=True,
                                output=ott_path)
        print(tt)
        tt.data.sort_values(by="event_index", inplace=True)
        tt.data["phase_hint"] = phase
        picks.append(tt.data)
    picks = pd.concat(picks)
    print(picks)
    
    # x,y,z,profiles = prepare_db1d_syn_vel_model(x,y,z,vel_path,proj)
    # stations = get_db_syn_stations(x,y,stations_path,proj)
    
    
    # for phase, vel in profiles.items():
    #     tt_path = os.path.join(tt_folder_path,f"{phase}_tt.h5")
    #     tt = EarthquakeTravelTimeDataset(phase,stations)
    #     tt.add_grid_with_velocity_model(x,y,z,nx,ny,nz,
    #                     xy_epsg=proj,
    #                     vel1d=vel)
    #     tt.save_traveltimes(output=tt_path)
    
    
    
    