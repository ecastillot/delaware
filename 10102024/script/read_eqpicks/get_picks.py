from delaware.core.read import EQPicks
import datetime as dt

root = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq"
author = "usgs_20170101_20240922"
proj = "EPSG:3857"
starttime = "2023-01-01 00:00:00"
endtime = "2023-02-01 00:00:00"

starttime = dt.datetime.strptime(starttime,
                                 "%Y-%m-%d %H:%M:%S")
endtime = dt.datetime.strptime(endtime,
                                 "%Y-%m-%d %H:%M:%S")

eqpicks = EQPicks(root=root,
                  author=author,
                  xy_epsg=proj,
                  catalog_header_line=1)
print(eqpicks.catalog)
catalog, picks = eqpicks.get_catalog_with_picks(
                                starttime=starttime,
                               endtime=endtime,
                               ev_ids=["texnet2023bjyx"])

print(catalog,picks)