# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2025-01-24 13:56:16
#  * @modify date 2025-01-24 13:56:16
#  * @desc [description]
#  */

from .spatial  import Points

class Stations(Points):
    def __init__(self, *args,**kwargs) -> None:
        mandatory_columns = ['sta_id', 'network', 'station', 
                            'latitude', 'longitude', 'elevation']
        super().__init__(*args,mandatory_columns = mandatory_columns,
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
    
    
    