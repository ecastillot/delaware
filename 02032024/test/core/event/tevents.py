from delaware.core.event.events import get_texnet_high_resolution_catalog
from delaware.core.event.picks import read_picks


events_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust/origin_r.csv"
events = get_texnet_high_resolution_catalog(events_path,xy_epsg="EPSG:3116",
                                            author="texnet")

# picks_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data/eq/aoi/growclust/picks.db"

# picks = read_picks(picks_path,author="texnet",
#                    ev_ids=["texnet2023abeo"])
# print(picks)
# picks = events.get_picks(picks_path)
# print(picks)


# events["ev_id"] = 1
# print(events)
# events = Events(events, xy_epsg="EPSG:3116", author=None,
#                 )
# print(events)

