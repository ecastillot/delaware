# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2025-01-24 13:56:16
#  * @modify date 2025-01-24 13:56:16
#  * @desc [description]
#  */

from .spatial  import Points
from .picks import Picks, read_picks
import pandas as pd

from datetime import datetime

def get_texnet_high_resolution_catalog(path,xy_epsg,author,
                                       depth_lims=[0,20],
                                        region_lims=None):
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
                            "EventId":"ev_id"})
    
    catalog = Events(df,xy_epsg=xy_epsg,author=author)
    
    if depth_lims is not None:
        catalog.filter("depth",start=depth_lims[0],end=depth_lims[1])
    
    # region_lims #lonw,lone,lats,latn
    if region_lims is not None:
        catalog.filter_rectangular_region(region_lims)
    
        
    catalog.sort_values(by="origin_time")
    return catalog

class Events(Points):
    def __init__(self, *args,**kwargs) -> None:
        required_columns = ['ev_id','origin_time','latitude',
                            'longitude','depth','magnitude']
        if "date_columns" not in list(kwargs.keys()):
            kwargs["date_columns"] = ["origin_time"]
        super().__init__(*args,required_columns = required_columns,
                         **kwargs)
        
    def __str__(self,extended=False) -> str:
        if extended:
            timefmt = "%Y%m%dT%H:%M:%S"
            start=  self.data.origin_time.min()
            end = self.data.origin_time.max()
            region = list(map(lambda x: round(x,2),self.get_region()))
            msg = f"Catalog | {self.__len__()} events "\
                    +f"\n\tperiod: [{start.strftime(timefmt)} - {end.strftime(timefmt)}]"\
                    +f"\n\tdepth : {[round(self.data.depth.min(),2),round(self.data.depth.max(),2)]}"\
                    +f"\n\tmagnitude : {[round(self.data.magnitude.min(),2),round(self.data.magnitude.max(),2)]}"\
                    +f"\n\tregion: {region}"
        else:
            msg = f"Catalog | {self.__len__()} events "

        return msg
    
    def query(self,starttime=None,
                endtime=None,
                ev_ids=None,agencies=None,
                mag_lims=None,region_lims=None,
                general_region=None,
                region_from_src=None):
        
        self.filter("origin_time",starttime,endtime)
        
        if (ev_ids is not None) and (len(self) !=0):
            self.select_data({"ev_id":ev_ids})
            
        if (agencies is not None) and (len(self) !=0):
            self.select_data({"agency":agencies}) #agencies is a list
        
        if (mag_lims is not None) and (len(self) !=0):
            self.filter("magnitude",start=mag_lims[0],
                               end=mag_lims[1])
        
        if (region_lims is not None) and (len(self) !=0):
            self.filter_rectangular_region(region_lims)
            
        if (general_region is not None) and (len(self) !=0):
            self.filter_general_region(general_region)
            
        if (region_from_src is not None) and (len(self) !=0):
            lat,lon, r_max, az_max =  region_from_src
            self.filter_by_r_az(latitude=lat,
                            longitude=lon,
                            r=r_max,
                            az=az_max)
        
        return self  
    
    def get_picks(self, picks_path,ev_ids=None,
                  stations =None,author=None):
        
        if len(self) != 0:
            default_ev_ids = self.data["ev_id"].to_list()
            if ev_ids is None:
                ev_ids = default_ev_ids
            else:
                ev_ids = [ev_id for ev_id in ev_ids if ev_id in default_ev_ids]
            
            if not ev_ids:
                raise Exception(f"No events found. Check your list of ids {ev_ids}")
            
        else :
            raise Exception(f"No events found. Your Events object is empty.")
       
        picks = read_picks(picks_path,author=author,
                               ev_ids=ev_ids,mode="utdquake")
        
        if picks.empty:
            return picks
       
        if stations is not None:
            stations_data = stations.data.copy()
            picks_data = picks.data.copy()
            renaming = {"latitude":"station_latitude",
                        "longitude":"station_longitude",
                        "elevation":"station_elevation"}
            to_rename = {k: v for k, v in renaming.items() \
                                    if v not in stations_data.columns}
            stations_data = stations_data.rename(columns=to_rename)
            pick_columns =  picks_data.columns.to_list()
            for key in renaming.values():
                if key in pick_columns:
                     picks_data.drop(key,axis=1,inplace=True)
            
            picks_data = pd.merge( picks_data,stations_data,
                            on=["network","station"],
                            )
        
            picks = Picks( picks_data,author=author)
        
        
        return  picks