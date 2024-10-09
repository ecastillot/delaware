import os
import pandas as pd
from SeisMonitor.monitor.locator.nlloc.nlloc import NLLoc
from SeisMonitor.monitor.locator import utils as lut
from delaware.eqviewer.utils import inside_the_polygon

out_dir = "/home/emmanuel/ecastillo/dev/delaware/21092024/output/nlloc_test_08102024"
vel_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/vel/DB_model.csv"
stations_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/stations/delaware_onlystations_160824.csv"
nlloc_path = "/home/emmanuel/NLLoc"
aoi = [-104.84329,-103.79942,31.39610,31.91505,-2,20]

vel_model = pd.read_csv(vel_path)
vel_model["Depth (km)"].replace("DTM", -0.750, inplace=True)
# depth	vp	vs	disc	rho
vel_model = vel_model.rename(columns={"Depth (km)":"depth","VP (km/s)":"vp", 
                          "VS (km/s)":"vs"})
vel_model["rho"] = 2.5
vel_model["depth"] = vel_model["depth"].astype(float)
vel_model = lut.VelModel(vel_model,model_name="DB1D",compute_vs=False)

stations = pd.read_csv(stations_path)

polygon = [(aoi[0],aoi[2]),
                (aoi[0],aoi[3]),
                (aoi[1],aoi[3]),
                (aoi[1],aoi[2]),
                (aoi[0],aoi[2])
                ]
is_in_polygon = lambda x: inside_the_polygon((x.longitude,x.latitude),polygon)
mask = stations[["longitude","latitude"]].apply(is_in_polygon,axis=1)
stations = stations[mask]
stations.reset_index(inplace=True)
stations = lut.Stations(stations)

tmp_nlloc_folder = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/nlloc"
nlloc = NLLoc(
        core_path = nlloc_path, ### type your NLLoc path, 
        agency="SeisMonitor",
        region = aoi,
        # region = [-85, -68,0, 15,-5, 205],
        vel_model = vel_model,
        stations = stations,
        delta_in_km = 1,
        tmp_folder=tmp_nlloc_folder### CHANGE PATH TO YOUR OWN PATH AND ALSO TAKE IN MIND THAT CONSUME DISK
        )

catalog = "/home/emmanuel/ecastillo/dev/delaware/21092024/output/nlloc_test_08102024/test_cat.xml"
output= "test_nlloc_cat.xml"
nlloc_catalog = nlloc.locate(catalog=catalog,
                            nlloc_out_folder= out_dir,
                            out_filename = "LOC.xml",
                            out_format="SC3ML" )
print(nlloc_catalog)