# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-09-22 22:07:55
#  * @modify date 2024-09-22 22:07:55
#  * @desc [description]
#  */

import os 
import glob
from datetime import timedelta
from obspy.core.util.misc import BAND_CODE
from obspy.clients.filesystem.sds import Client


# Only for waveforms!
class LocalClient(Client):

    def __init__(self,root,fmt,**kwargs):
        """
        This script is an example to make a client class
        for specific data structure archive on local filesystem. 

        The mandatory parameters for LocalClient class is: root_path and field_name
        Example:
        ---------
        root = "/home/emmanuel/myarchive"
        fmt = "{year}-{month:02d}/{year}-{month:02d}-{day:02d}/{network}.{station}.{location}.{channel}.{year}.{julday:03d}"
        client = LocalClient(root,fmt)
        st = client.get_waveforms("YY","XXXX","00",
                                channel="HHZ",starttime = UTCDateTime("20220102T000100"),
                                endtime = UTCDateTime("20220102T000200"))
        
        Parameters:
        -----------
        root: str
            Path where is located the Local structure
        fmt: str
            The parameter should name the corresponding keys of the stats object, e.g. 
            "{year}-{month:02d}/{year}-{month:02d}-{day:02d}/{network}.{station}.{location}.{channel}.{year}.{julday:03d}"

        **kwargs SDS client additional args
        """
        self.root = root
        self.fmt = fmt
        super().__init__(root,**kwargs)

    def _get_filenames(self, network, station, location, channel, starttime,
                       endtime, sds_type=None):
        """
        Get list of filenames for certain waveform and time span.
        :type network: str
        :param network: Network code of requested data (e.g. "IU").
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
        :type location: str
        :param location: Location code of requested data (e.g. "").
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Time of interest.
        :type sds_type: str
        :param sds_type: None
        :rtype: str
        """
        sds_type = sds_type or self.sds_type
        # SDS has data sometimes in adjacent days, so also try to read the
        # requested data from those files. Usually this is only a few seconds
        # of data after midnight, but for now we play safe here to catch all
        # requested data (and with MiniSEED - the usual SDS file format - we
        # can use starttime/endtime kwargs anyway to read only desired parts).
        year_doy = set()
        # determine how far before starttime/after endtime we should check
        # other dayfiles for the data
        t_buffer = self.fileborder_samples / BAND_CODE.get(channel[:1], 20.0)
        t_buffer = max(t_buffer, self.fileborder_seconds)
        t = starttime - t_buffer
        t_max = endtime + t_buffer
        # make a list of year/doy combinations that covers the whole requested
        # time window (plus day before and day after)
        while t < t_max:
            year_doy.add((t.year,t.month,t.day, t.julday))
            t += timedelta(days=1)
        year_doy.add((t_max.year,t_max.month,t_max.day, t_max.julday))

        full_paths = set()
        for year,month,day,doy in year_doy:
            filename = self.fmt.format(
                            network=network, station=station, location=location,
                            channel=channel, year=year, month=month, 
                            day=day, julday=doy,sds_type=sds_type)
            full_path = os.path.join(self.sds_root, filename)
            full_paths = full_paths.union(glob.glob(full_path))
        
        return full_paths

    def _get_filename(self, network, station, location, channel, time, sds_type=None):
        """
        Get filename for certain waveform.
        :type network: str
        :param network: Network code of requested data (e.g. "IU").
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
        :type location: str
        :param location: Location code of requested data (e.g. "").
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Time of interest.
        """
        sds_type = sds_type or self.sds_type
        filename = self.fmt.format(
                    network=network, station=station, location=location,
                    channel=channel, year=time.year, month=time.month, 
                    day=time.day, doy=time.julday,sds_type=sds_type)
        return os.path.join(self.sds_root, filename)