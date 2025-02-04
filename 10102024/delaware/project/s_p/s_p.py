import os
import tqdm 
from delaware.project.datatools import *

def get_events_by_sp(station,catalog,picks,
                    rmax,zmin=None,
                    output_folder=None):
        
        if output_folder is not None:
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
        
        sta_data = station.drop_duplicates("station",ignore_index=True)
        
        # events = {}
        all_events = dict((sta,[]) for sta in sta_data["station"].to_list())
        all_picks = dict((sta,[]) for sta in sta_data["station"].to_list())
        
        # Wrap the iterrows() with tqdm for progress bar
        for i, row in tqdm(sta_data.iterrows(), total=len(sta_data),
                           desc="Processing stations"):
            new_catalog = catalog.copy()
            if zmin is not None:
                new_catalog = filter_data(new_catalog,"depth", start=zmin)
            
            new_catalog = filter_data_by_r_az(new_catalog,
                                         latitude=row.latitude,
                                        longitude=row.longitude,
                                        r=rmax)
            if new_catalog.empty:
                continue
            
            if picks_path is not None:
                picks = new_catalog.get_picks(picks_path)
                picks.select_data({"station":[row.station]})
                picks.filter_requiring_ps_phases_in_station()
                
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
                
                save_dataframe_to_sqlite(new_catalog.data,
                                            db_name=sp_cat_path,
                                        table_name=row.station)
                
                if picks_path is not None:
                    save_dataframe_to_sqlite(picks.data,
                                            db_name=sp_picks_path,
                                        table_name=row.station)
            # new_catalog = Catalog(data=new_catalog,xy_epsg=self.xy_epsg)        
            all_events[row.station].append(new_catalog)
            
            if picks_path is not None:
                all_picks[row.station].append(picks)
            
        
        return all_events,all_picks