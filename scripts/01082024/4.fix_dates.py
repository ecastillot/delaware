import pandas as pd
from datetime import datetime

path = "/mnt/Ecopetrol/Analytics/emmanuel/dev/others/texnet_hirescatalog.csv"
outpath = "/home/emmanuel/ecastillo/dev/delaware/data/metadata/texnet_hirescatalog_fixed.csv"
df = pd.read_csv(path)
# Function to create datetime
def create_datetime(row):
    try:
        return datetime(year=row['yr'], month=row['mon'], day=row['day'],
                        hour=row['hr'], minute=row['min_'], second=int(row['sec']),
                        microsecond=int((row['sec'] % 1) * 1e6))
    except ValueError:
        return pd.NaT  # or use `None` if you want to see errors

# Apply function to create 'origin_time'
df['origin_time'] = df.apply(create_datetime, axis=1)
df.to_csv(outpath,index=False)