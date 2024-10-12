
if __name__ == "__main__":
    import os
    from delaware.utils import *
    from delaware.synthetic.tt import get_picks
    from delaware.core.eqviewer import Catalog
    
    x = (-104.84329,-103.79942)
    y = (31.3961,31.91505)
    z = (-2,10)
    nx, ny, nz = 30,40,60
    proj = "EPSG:3857"
    stations_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/stations/standard_stations.csv"
    tt_folder_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/synthetics/tt"
    eq_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/growclust/texnet_hirescatalog.csv"
    
    stations = get_db_stations(stations_path,x,y,proj)
    print(stations)
    catalog = get_texnet_high_resolution_catalog(eq_path,xy_epsg=proj,
                                                 region_lims=x+y)
    
    print(catalog)
    # exit()
    
    # "event_test"
    # df = pd.DataFrame.from_dict({"latitude":[31.636749999999999],
    #                          "longitude":[-103.998790000000000],
    #                         #  "depth":[6.903000000000000],
    #                          "depth":[2.5000000000000],
    #                          "magnitude":[1],
    #                         #  "origin_time":[dt.datetime(2022,11,16,21,32,48.4819999)]
    #                          "origin_time":["2022-11-16 21:32:48.4819999"],
    #                          "ev_id":["edep"]
    #                          })
    # df["origin_time"] = pd.to_datetime(df['origin_time'])
    # catalog = Catalog(data=df, xy_epsg=proj)
    
    
    output_folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/pykonal_growclust"
    
    eq= get_picks(tt_folder_path=tt_folder_path,
                            stations=stations,
                            catalog=catalog,
                            xy_epsg=proj,
                            output_folder=output_folder,
                            join_catalog_id=True)
    # print(eq)
    
    
    
    
    