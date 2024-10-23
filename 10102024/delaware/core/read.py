# read earthquakes and picks
import os
import copy
import pandas as pd
from delaware.core.eqviewer import Catalog,MulPicks
from delaware.core.eqviewer_utils import get_distance_in_dataframe
from delaware.core.database import save_dataframe_to_sqlite

class EQPicks():
    def __init__(self,root,author,xy_epsg,catalog_header_line=0,
                 ):
        self.root = root
        self.author = author
        
        picks_path = os.path.join(root,author,"picks.db")
        catalog_path = os.path.join(root,author,"origin.csv")
        
        for path in [picks_path,catalog_path]:
            if not os.path.isfile(path):
                raise Exception(f"There is not {path}")
        
        self.picks_path = picks_path
        self.catalog_path = catalog_path
        self.xy_epsg = xy_epsg
        self.catalog_header_line = catalog_header_line
        self.catalog = self._get_catalog()
    
    def _get_catalog(self):
        catalog = pd.read_csv(self.catalog_path,parse_dates=["origin_time"],
                              header=self.catalog_header_line)
        catalog = catalog.drop_duplicates(subset=["ev_id"],ignore_index=True)
        
        if "magnitude" not in catalog.columns.to_list():
            catalog["magnitude"] = 1 #due to pykonal database
            
        catalog = Catalog(catalog,xy_epsg=self.xy_epsg)
        return catalog
    
    def copy(self):
        """Deep copy of the class"""
        return copy.deepcopy(self)
    
    def query(self,**kwargs):
        self.catalog.query(**kwargs)
        return self
    
    def select_data(self,rowval):
        self.catalog.select_data(rowval=rowval)
        return self
    
    def get_catalog_with_picks(self,starttime=None,
                               endtime=None,
                               ev_ids=None,
                               mag_lims=None,region_lims=None,
                               general_region=None,
                               region_from_src=None,
                               stations = None):
        
        for query in [ev_ids,mag_lims,region_lims]:
            if query is not None:
                if not isinstance(query,list):
                    raise Exception(f"{query} must be a list")
        
        new_catalog = self.catalog.copy()
        
        picks = new_catalog.get_picks(picks_path=self.picks_path,
                                      ev_ids=ev_ids,
                                      starttime=starttime,
                                      endtime=endtime,
                                      general_region=general_region,
                                      region_lims=region_lims,
                                      region_from_src=region_from_src,
                                      author=self.author,
                                      stations=stations)
        
        if stations is not None:
            cat_info = new_catalog.data.copy()[["ev_id","latitude","longitude"]]
            cat_columns = {"latitude":"src_latitude","longitude":"src_longitude"}
            cat_info = cat_info.rename(columns=cat_columns)
            picks_data = picks.data        
            picks_data = pd.merge(picks_data,cat_info,on=["ev_id"])
            # print(picks_data.columns)
            picks_data = get_distance_in_dataframe(data=picks_data,lat1_name="src_latitude",
                                          lon1_name="src_longitude",
                                          lat2_name="station_latitude",
                                          lon2_name="station_longitude",
                                          columns=["sr_r [km]",
                                                   "sr_az","sr_baz"])
            picks.data = picks_data
        
        return new_catalog, picks


class ParseEQPicks():
    def __init__(self,eqpicks1,eqpicks2) -> None:
        self.eqpicks1 = eqpicks1
        self.eqpicks2 = eqpicks2
        
    def compare(self,ev_ids=None,out=None,**kwargs):
        
        if ev_ids is None:
            ev_ids = list(set(self.eqpicks1.catalog.data["ev_id"].to_list())) 
        
        kwargs["ev_ids"] = ev_ids 
        
        ## filtering data
        query_kwargs = kwargs.copy()
        query_kwargs.pop("stations",None)
        all_eqpicks = []
        for _eqpicks in [self.eqpicks1,self.eqpicks2]:
            eqpicks = _eqpicks.copy()
            eqpicks.query(**query_kwargs)
            all_eqpicks.append(eqpicks)
        
        # getting picks
        all_comparisons = []
        for ev_id in ev_ids:
            
            all_picks = {}
            for _eqpicks in all_eqpicks:
                eqpicks = _eqpicks.copy() #important
                eqpicks.select_data({"ev_id":[ev_id]})
                try:
                    catalog, picks = eqpicks.get_catalog_with_picks(ev_ids=kwargs["ev_ids"],
                                                                    stations=kwargs["stations"])
                except:
                    print(f"Event [BAD]: {ev_id} | Author {eqpicks.author}: get_catalog_with picks didn't work ")
                    continue
                all_picks[picks.author] = picks
            
            if len(all_picks) < 2:
                print(f"\tEvent skipped: {ev_id} ")
                continue
            
            #comparing
            mulpicks = MulPicks(list(all_picks.values()))
            comparison = mulpicks.compare(*list(all_picks.keys()))
            
            print(f"Event [OK]: {ev_id} ")
            if out is not None:
                if not os.path.isdir(os.path.dirname(out)):
                    os.makedirs(os.path.dirname(out))
                save_dataframe_to_sqlite(comparison, out, ev_id)
            else:
                all_comparisons.append(comparison)
        
        if all_comparisons:
            all_comparisons = pd.concat(all_comparisons)
        return all_comparisons
            # print(comparison)
        
            
        
    # def write_catalog_with_picks(self,**kwargs):
    #     new_catalog = self.catalog.copy()
    #     new_catalog.query(**kwargs)
    #     catalog_data = new_catalog.data
        
    #     groupby = catalog_data.groupby("ev_id")
        
    #     for ev_id, data in groupby.__iter__():
    #         single_catalog = Catalog(data=data,
    #                                  xy_epsg=self.catalog.xy_epsg)
            
    #         single_catalog, picks = single_catalog.get_picks(picks_path=self.picks_path,
    #                                             ev_ids=[ev_id],
    #                                             )
            
    #         print(single_catalog, picks )
        
    #     return None            
        
        
        