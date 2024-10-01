from delaware.loc.s_p import SP_Database

# catalog_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/nlloc/catalog_sp_method.db"
# picks_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/nlloc/picks_sp_method.db"

catalog_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/growclust/catalog_sp_method.db"
picks_path = "/home/emmanuel/ecastillo/dev/delaware/21092024/data/loc/s_p/growclust/picks_sp_method.db"


sp_db = SP_Database(catalog_path=catalog_path,
                    picks_path=picks_path)

print(sp_db)
print(sp_db.n_stations)
print(sp_db.n_events)
sp_db.plot_stations_counts()