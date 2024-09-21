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

def plot_traces(stream,picks_list,out=None,show=False,figsize=(12,12),fontsize=10):
    fig, ax = plt.subplots(len(stream), sharex=True,gridspec_kw = {'wspace':0, 'hspace':0},figsize=figsize)
    
    color_authors = {"eqt":{"P":"blue",
                            "S":"red"},
                     "pykonal":{"P":"cyan",
                                "S":"magenta"},
                     "pykonal_2.5km":{"P":"green",
                                "S":"purple"}
                     
                     }
    
    
    all_labels = []
    for i,tr in enumerate(stream):
        #print(tr.id)
        start = tr.stats.starttime.datetime
        end = tr.stats.endtime.datetime
        deltadt = (end-start).total_seconds()
        delta = tr.stats.delta
        x =[start + dt.timedelta(seconds=x) for x in np.arange(0,deltadt+delta,delta)]
        
        for author,picks in picks_list.items():
            ### ps_picks section
            ps_picks = picks[(picks["arrival_time"]>=start) &\
                            (picks["arrival_time"]<=end)]
            ps_picks = ps_picks[ps_picks["station"] == tr.stats.station]
            
            if not ps_picks.empty:
                for l,pick in ps_picks.iterrows():
                    pick_time = pick["arrival_time"]
                    # pick_time = pick_time - start
                    
                    ymin, ymax = ax[i].get_ylim()
                    
                    phasehint = pick["phasehint"]
                    
                    label = f"{phasehint}-{author.capitalize()}"
                    
                    if label not in all_labels: 
                        phase_label = label
                        all_labels.append(label)
                    else:
                        phase_label=None
                        
                    ax[i].vlines(pick_time , ymin, ymax, 
                                        color=color_authors[author][phasehint],
                                        label=phase_label,
                                        linewidth=2)
                    
                    
            #print(deltadt)
            ax[i].plot(x,tr.data, 'k')
            ax[i].set_yticklabels([])
            ax[i].text(0.05, 0.95, f'{tr.id}', 
                        transform=ax[i].transAxes, fontsize=fontsize,
                        style='italic',color="red", bbox={
                        'facecolor': 'white', 'alpha': 0.5, 'pad': 2})

            startformat = tr.stats.starttime.strftime("%Y-%m-%d %H:%M:%S.%f")
            ax[i].set_xlabel(f"Time", fontsize=14)
        
    fig.legend(loc="upper right",ncol=4,bbox_to_anchor=(1, 1) )    
    fig.autofmt_xdate()
    # plt.tight_layout() 
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust right margin to fit the legend 
    
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
    import datetime as dt

    stations = pd.read_csv(r"/home/emmanuel/ecastillo/dev/delaware/data/metadata/delaware_onlystations_160824.csv")
    # st = read(r"\\esg.net\datashare\ISM\Ecopetrol\ECP_questions\AllNetworks_2023_03_10_09_10\st_CA_CH\CA_CH_2023_03_10_09_10.mseed")
    # info = st._get_common_channels_info()
    # stations = list(zip(*list(info.keys())))
    # stations = list(stations[1])
    # print(stations)
    # print(list(st.keys()))
    # exit()
    st = r"/home/emmanuel/ecastillo/dev/delaware/monitor2/downloads"
    
    picks = "/home/emmanuel/ecastillo/dev/delaware/monitor2/picks/eqt/seismonitor_picks.csv"
    picks = pd.read_csv(picks,parse_dates=["arrival_time"])
    
    # syn_picks = "/home/emmanuel/ecastillo/dev/delaware/monitor2/picks/pykonal/single_eq.csv"
    # syn_picks = pd.read_csv(syn_picks)
    # origin_time = dt.datetime.strptime("2022-11-16 21:32:44.481999","%Y-%m-%d %H:%M:%S.%f")
    # syn_picks["arrival_time"] = syn_picks["travel_time"].apply(lambda x: dt.timedelta(seconds=x)+origin_time)
    # # print(syn_picks)
    # # exit()
    
    
    syn_picks = "/home/emmanuel/ecastillo/dev/delaware/monitor2/picks/pykonal/single_eq2.csv"
    syn_picks = pd.read_csv(syn_picks)
    origin_time = dt.datetime.strptime("2023-12-07 04:05:49.408999","%Y-%m-%d %H:%M:%S.%f")
    syn_picks["arrival_time"] = syn_picks["travel_time"].apply(lambda x: dt.timedelta(seconds=x)+origin_time)
    # print(syn_picks)
    
    st = read_st(st,leaving_gaps=False,files_format=".mseed")
    print(st)
    st = st.select(component="Z")
    # st = st.select(component="N",station="CA02A")
    st = st.trim(starttime=UTCDateTime("2023-12-07T04:05:30.000000"),
                    endtime=UTCDateTime("2023-12-07T04:06:30.000000"),)
    # st.plot(method="full")
    # st = st.merge(method=1)
    # print(st)
    # for tr in st:
    #     if isinstance(tr.data, np.ma.masked_array):
    #         tr.data = tr.data.filled()
    # # # st = st.split()

    
    myst = MyStream(st.traces,stations)
    myst.sort_from_source(source=(-104.050,31.64771000))
    myst.detrend().normalize()
    # myst.normalize()
    # plot_traces(myst,picks)
    
    picks_list = {"eqt":picks,"pykonal":syn_picks,
                #   "pykonal_2.5km":syn2_picks,
                  }
    plot_traces(myst,picks_list)
    plt.show()
