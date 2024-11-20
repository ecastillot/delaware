from delaware.core.eqviewer import Catalog
from delaware.utils import get_texnet_high_resolution_catalog
import pandas as pd

x = (-104.84329,-103.79942)
y = (31.3961,31.91505)
z = (-2,20)
proj = "EPSG:3857"

c1 = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/clusters/C3.bna"

c1 =pd.read_csv(c1)
c1 = list(zip(c1['lon'], c1['lat']))
# print(x)

#growclust catalog
# cat_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/growclust/texnet_hirescatalog.csv"
cat_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/eq/aoi/growclust/texnet_hirescatalog.csv"
out_cat_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/clusters/eq_c3.csv"
catalog = get_texnet_high_resolution_catalog(cat_path,xy_epsg=proj,
                                             region_lims=x+y,
                                             depth_lims=z)
catalog.filter_general_region(c1)
print(catalog)
catalog.data.to_csv(out_cat_path,index=False)

#nlloc catalog
# cat_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/growclust/texnet_hirescatalog.csv"
# out_cat_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/usgs_20170101_20240922/origin.csv"
# data = pd.read_csv(cat_path,parse_dates=["origin_time"],header=1)
# data["depth"] /= 1e3
# catalog = Catalog(data,xy_epsg=proj)
# catalog.filter_rectangular_region(x+y)
# catalog.filter("depth",z[0],z[1])
# catalog.data.to_csv(out_cat_path,index=False)