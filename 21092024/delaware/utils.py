# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-09-24 21:02:12
#  * @modify date 2024-09-24 21:02:12
#  * @desc [description]
#  */
import pandas as pd
from datetime import datetime
from delaware.eqviewer.eqviewer import Catalog

def get_texnet_high_resolution_catalog(path):
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
    df = df.rename(columns={"latR":"latitude",
                            "lonR":"longitude",
                            "depR":"depth",
                            "mag":"magnitude",
                            "EventId":"id"})
    
    catalog = Catalog(df)
    return catalog
    
    # df.to_csv(outpath,index=False)
    
