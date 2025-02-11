# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2025-01-24 13:56:16
#  * @modify date 2025-01-24 13:56:16
#  * @desc [description]
#  */

from .spatial  import Points
from ..database.database import save_to_sqlite
from tqdm import tqdm
import os

class Stations(Points):
    def __init__(self, *args,**kwargs) -> None:
        required_columns = ['sta_id', 'network', 'station', 
                            'latitude', 'longitude', 'elevation']
        super().__init__(*args,required_columns = required_columns,
                         **kwargs)
        self.data["z[km]"] = self.data["elevation"]/1e3 * -1
        
    def __str__(self,extended=False) -> str:
        msg = f"Stations | {self.__len__()} stations"
        if extended:
            region = list(map(lambda x: round(x,2),self.get_region()))
            msg += f"\n\tregion: {region}"
        else:
            pass
        return msg
    
    def get_events_by_sp(self,catalog,rmax,
                         zmin=None,
                         parse_dates=None,
                         picks_path=None,
                         output_folder=None):
        
        if output_folder is not None:
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
        
        sta_data = self.data.drop_duplicates("station",ignore_index=True)
        
        # events = {}
        all_events = dict((sta,[]) for sta in sta_data["station"].to_list())
        all_picks = dict((sta,[]) for sta in sta_data["station"].to_list())
        
        # Wrap the iterrows() with tqdm for progress bar
        for i, row in tqdm(sta_data.iterrows(), total=len(sta_data),
                           desc="Processing stations"):
            new_catalog = catalog.copy()
            if zmin is not None:
                new_catalog.filter("depth", start=zmin)
            new_catalog.filter_by_r_az(latitude=row.latitude,
                                    longitude=row.longitude,
                                    r=rmax)
            # print(row.station,row.latitude,row.longitude,new_catalog)
            
            if new_catalog.empty:
                # print("bad",i,new_catalog)
                continue
            
            if picks_path is not None:
                
                picks = new_catalog.get_picks(picks_path,stations=self,
                                parse_dates=parse_dates)
                picks.select_data({"station":[row.station]})
                picks.drop_picks_with_single_phase(inplace=True)
                
                if picks.empty:
                    continue
                
                ev_ids = picks.data["ev_id"].to_list()
                new_catalog.select_data(rowval={"ev_id":ev_ids})
                if new_catalog.empty:
                    continue
        
            if output_folder is not None:
                
                sp_cat_path = os.path.join(output_folder,
                                            "catalog_sp_method.db")
                sp_picks_path = os.path.join(output_folder,
                                                "picks_sp_method.db")
                
                save_to_sqlite(new_catalog.data,
                                db_path=sp_cat_path,
                                table_name=row.station)
                
                if picks_path is not None:
                    save_to_sqlite(picks.data,
                                    db_path=sp_picks_path,
                                    table_name=row.station
                                    )
        
        # new_catalog = Catalog(data=new_catalog,xy_epsg=self.xy_epsg)        
            all_events[row.station].append(new_catalog)
            
            if picks_path is not None:
                all_picks[row.station].append(picks)
        return all_events,all_picks
        
    
    
    