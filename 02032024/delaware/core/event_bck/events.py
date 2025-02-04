# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2025-01-24 18:56:41
#  * @modify date 2025-01-24 18:56:41
#  * @desc [description]
#  */

from .spatial  import Points
from .picks import read_picks

class Events(Points):
    """
    A class representing a collection of seismic events.

    Inherits from:
    --------------
    Points : Base class for handling geospatial point data.

    Attributes:
    -----------
    mandatory_columns : list
        Required columns for the event data: ['ev_id', 'origin_time', 'latitude', 'longitude', 'depth', 'magnitude'].
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Events class with mandatory and optional attributes.

        Parameters:
        -----------
        *args : tuple
            Positional arguments to be passed to the parent class.
        **kwargs : dict
            Keyword arguments, including optional configurations for the parent class.
        """
        # Define the mandatory columns for the Events data
        mandatory_columns = ['ev_id', 'origin_time', 
                             'latitude', 'longitude', 'depth', 'magnitude']
        
        # Call the parent class constructor with the required and optional parameters
        super().__init__(*args, 
                         mandatory_columns=mandatory_columns,
                         date_columns=["origin_time"],
                         **kwargs)

    def __str__(self, extended=False) -> str:
        """
        String representation of the Events class.

        Parameters:
        -----------
        extended : bool, optional
            Whether to include extended details about the events (default is False).

        Returns:
        --------
        str
            A formatted string summarizing the event data.
        """
        if extended:
            # Define the time format for display
            timefmt = "%Y%m%dT%H:%M:%S"

            # Get the time range, region, depth, and magnitude summary
            start = self.data.origin_time.min()
            end = self.data.origin_time.max()
            region = list(map(lambda x: round(x, 2), self.get_region()))
            
            # Prepare the detailed message
            msg = (
                f"Events | {self.__len__()} events "
                f"\n\tperiod: [{start.strftime(timefmt)} - {end.strftime(timefmt)}]"
                f"\n\tdepth : [{round(self.data.depth.min(), 2)}, {round(self.data.depth.max(), 2)}]"
                f"\n\tmagnitude : [{round(self.data.magnitude.min(), 2)}, {round(self.data.magnitude.max(), 2)}]"
                f"\n\tregion: {region}"
            )
        else:
            # Prepare a short summary
            msg = f"Events | {self.__len__()} events "

        return msg

    def query(self, starttime=None, endtime=None, ev_ids=None, agencies=None,
              mag_lims=None, region_lims=None, general_region=None,
              region_from_src=None):
        """
        Query and filter events based on various criteria.

        Parameters:
        -----------
        starttime : datetime, optional
            Start time for filtering events (default is None).
        endtime : datetime, optional
            End time for filtering events (default is None).
        ev_ids : list, optional
            List of event IDs to include (default is None).
        agencies : list, optional
            List of agencies to include (default is None).
        mag_lims : tuple, optional
            Magnitude range as (min, max) (default is None).
        region_lims : list, optional
            Rectangular region limits [lonw,lone,lats,latw] (default is None).
        general_region : list of tuples
            Each tuple is consider a point (lon,lat).
            The first point must be equal to the last point in the polygon.
        region_from_src : tuple, optional
            Source-based region definition (latitude, longitude, max_radius_in_km, max_azimuth) (default is None).

        Returns:
        --------
        self : Events
            The filtered Events object.
        """
        # Filter by origin time range
        self.filter("origin_time", starttime, endtime)

        # Filter by event IDs, if provided
        if ev_ids is not None and len(self) != 0:
            self.select_data({"ev_id": ev_ids})

        # Filter by agencies, if provided
        if agencies is not None and len(self) != 0:
            self.select_data({"agency": agencies})  # 'agencies' is a list

        # Filter by magnitude limits, if provided
        if mag_lims is not None and len(self) != 0:
            self.filter("magnitude", start=mag_lims[0], end=mag_lims[1])

        # Filter by rectangular region limits, if provided
        if region_lims is not None and len(self) != 0:
            self.filter_rectangular_region(region_lims)

        # Filter by general region, if provided
        if general_region is not None and len(self) != 0:
            self.filter_general_region(general_region)

        # Filter by source-based region, if provided
        if region_from_src is not None and len(self) != 0:
            lat, lon, r_max, az_max = region_from_src
            self.filter_by_r_az(latitude=lat, longitude=lon, r=r_max, az=az_max)

        return self
    
    # def get_picks(self,picks_path,author,stations=None,**kwargs):
        
    #     self.query(**kwargs)
    #     ev_ids = self.data["ev_id"].to_list()
    #     picks = read_picks(picks_path,author,ev_ids=ev_ids)
        
    #     # if stations is not None:
            
        
    #     return picks
    