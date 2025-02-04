import pandas as pd
from delaware.core.database.database import load_from_sqlite

picks_path = "/home/emmanuel/ecastillo/dev/delaware/02032024/project/sp/data/picks_sp_method.db"
picks = load_from_sqlite(picks_path)
print(picks.info())



