# read earthquakes and picks
import os
import pandas as pd
from delaware.core.eqviewer import Catalog

class EQPicks():
    def __init__(self,root,author,xy_epsg,catalog_header_line=0):
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
        catalog = Catalog(catalog,xy_epsg=self.xy_epsg)
        return catalog
        
    
    def get_catalog_with_picks(self,starttime,endtime,ev_ids=None,
                               mag_lims=None,region_lims=None,
                               general_region=None,
                               region_from_src=None):
        
        for query in [ev_ids,mag_lims,region_lims]:
            if query is not None:
                if not isinstance(query,list):
                    raise Exception(f"{query} must be a list")
        
        new_catalog = self.catalog.copy()
        
        picks = new_catalog.get_picks(picks_path=self.picks_path,
                                      event_ids=ev_ids,
                                      starttime=starttime,
                                      endtime=endtime,
                                      region_lims=region_lims,
                                      region_from_src=region_from_src)
        return new_catalog, picks
            
        
        
        
        