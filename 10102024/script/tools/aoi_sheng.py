from delaware.core.eqviewer import Catalog
from delaware.utils import get_texnet_high_resolution_catalog
import pandas as pd

x = (-104.84329,-103.0733)
y = (31.0434,31.91505)
z = (-2,20)
proj = "EPSG:3857"

#growclust catalog
cat_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/growclust/texnet_hirescatalog.csv"
out_cat_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust_and_sheng/origin.csv"
catalog = get_texnet_high_resolution_catalog(cat_path,xy_epsg=proj,
                                             region_lims=x+y,
                                             depth_lims=z)
print(catalog)
catalog.data.to_csv(out_cat_path,index=False)

#nlloc catalog
# cat_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/usgs_20170101_20240922/all_origin.csv"
# out_cat_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/usgs_20170101_20240922/origin.csv"
# data = pd.read_csv(cat_path,parse_dates=["origin_time"],header=1)
# data["depth"] /= 1e3
# catalog = Catalog(data,xy_epsg=proj)
# catalog.filter_rectangular_region(x+y)
# catalog.filter("depth",z[0],z[1])
# catalog.data.to_csv(out_cat_path,index=False)
