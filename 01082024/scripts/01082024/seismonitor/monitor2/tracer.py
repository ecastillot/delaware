import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth
from obspy.core.stream import Stream
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os

class MyStream(Stream):
    def __init__(self,traces,stations):
        """
        Parameters:
            stations: pd.Dataframe
            Dataframe with the station information
        """
        super().__init__(traces)
        self.stations = stations

    def get_stations(self):
        networks = []
        stations = []
        for tr in self.traces:
            stats = tr.stats
            stations.append(stats.station)
            networks.append(stats.network)
        stations = pd.DataFrame({"network":networks,
                                "station":stations})
        stations = pd.merge(self.stations,stations,how="inner")
        return stations


    def sort_from_source(self,source):
        """
        Parameters:
        source: tuple
            (lon,lat)

        returns: Stream
            New sorted stream
        """
        src_lon,src_lat = source
        stations = self.get_stations()
        distance_func = lambda y: gps2dist_azimuth(y.latitude,y.longitude,
                                                    src_lat,src_lon)[0]
        stations["r"] = stations.apply(distance_func,axis=1)
    
        #stations = stations.sort_values(by="r",ignore_index=True)
        stations = stations.sort_values(by="r")

        stream = Stream(self.traces)
        info = stream._get_common_channels_info()
        true_stations = list(zip(*list(info.keys())))
        true_stations = list(true_stations[1])

        sorted_traces = []
        for i,row in stations.iterrows():
            station = row.station
            if station in true_stations:
                st = stream.select(station=station)
                for tr in st:
                    sorted_traces.append(tr)
        self.traces = sorted_traces
        return self

def plot_traces(stream,out=None,show=False,figsize=(12,12),fontsize=10):
    fig, ax = plt.subplots(len(stream), sharex=True,gridspec_kw = {'wspace':0, 'hspace':0},figsize=figsize)
    for i,tr in enumerate(stream):
        #print(tr.id)
        start = tr.stats.starttime.datetime
        end = tr.stats.endtime.datetime
        deltadt = (end-start).total_seconds()
        delta = tr.stats.delta
        x =[start + dt.timedelta(seconds=x) for x in np.arange(0,deltadt+delta,delta)]

        #print(delta)
        #print(deltadt)
        ax[i].plot(x,tr.data, 'k')
        ax[i].set_yticklabels([])
        ax[i].text(0.05, 0.95, f'{tr.id}', 
                     transform=ax[i].transAxes, fontsize=fontsize,
                    style='italic',color="red", bbox={
                    'facecolor': 'white', 'alpha': 0.5, 'pad': 2})

        startformat = tr.stats.starttime.strftime("%Y-%m-%d %H:%M:%S.%f")
        ax[i].set_xlabel(f"time")
        
    fig.autofmt_xdate()
    plt.tight_layout()  
    if out != None:
        if not os.path.isdir(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
        fig.savefig(out)
        
    if show:
        plt.show()

    return fig,ax


    

if __name__ == "__main__":
    # from athena import AthenaClient
    from obspy import UTCDateTime,read,Stream
    import pandas as pd
    from read import read_st

    stations = pd.read_csv(r"/home/emmanuel/ecastillo/dev/delaware/data/metadata/delaware_onlystations_160824.csv")
    # st = read(r"\\esg.net\datashare\ISM\Ecopetrol\ECP_questions\AllNetworks_2023_03_10_09_10\st_CA_CH\CA_CH_2023_03_10_09_10.mseed")
    # info = st._get_common_channels_info()
    # stations = list(zip(*list(info.keys())))
    # stations = list(stations[1])
    # print(stations)
    # print(list(st.keys()))
    # exit()
    st = r"/home/emmanuel/ecastillo/dev/delaware/monitor/downloads"
    st = read_st(st,leaving_gaps=False,files_format=".mseed")
    print(st)
    st = st.select(component="Z")
    # st = st.select(component="N",station="CA02A")
    st = st.trim(starttime=UTCDateTime("2022-11-16T21:30:00.000000Z"),
                    endtime=UTCDateTime("2022-11-16T21:35:00.000000Z"),)
    # st.plot(method="full")
    # st = st.merge(method=1)
    # print(st)
    # for tr in st:
    #     if isinstance(tr.data, np.ma.masked_array):
    #         tr.data = tr.data.filled()
    # # # st = st.split()

    
    myst = MyStream(st.traces,stations)
    myst.sort_from_source(source=(-103.99,31.63))
    # myst.detrend().normalize()
    myst.normalize()
    plot_traces(myst)
    plt.show()
