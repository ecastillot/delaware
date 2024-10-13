import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth
from obspy.core.stream import Stream
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
from obspy.clients.fdsn import Client 
from obspy import UTCDateTime


class MyStream(Stream):
    """
    Custom class to extend ObsPy's Stream class with station-related functionality.
    
    Parameters:
        stations (pd.DataFrame): DataFrame containing station information.
        mandatory columns: 'network','station'
    """
    def __init__(self, traces, stations):
        super().__init__(traces)
        self.stations = stations

    def get_stations(self):
        """
        Extract network and station names from the trace stats and merge with
        the provided station DataFrame.

        Returns:
            pd.DataFrame: Merged DataFrame with station and network information.
        """
        networks = []
        stations = []
        # Collect network and station information from each trace
        for tr in self.traces:
            stats = tr.stats
            stations.append(stats.station)
            networks.append(stats.network)
        
        # Create DataFrame of network and station names
        stations = pd.DataFrame({"network": networks, "station": stations})
        
        # Merge with the provided station information
        stations = pd.merge(self.stations, stations, how="inner")
        
        return stations

    def sort_from_source(self, source):
        """
        Sort the traces in the stream based on their distance from a given source location.

        Parameters:
            source (tuple): Tuple representing the source coordinates (lon, lat).

        Returns:
            MyStream: Stream with traces sorted by distance from the source.
        """
        src_lon, src_lat = source
        
        # Get station information and calculate distances from the source
        stations = self.get_stations()
        distance_func = lambda y: gps2dist_azimuth(y.latitude, y.longitude, 
                                                   src_lat, src_lon)[0]
        stations["r"] = stations.apply(distance_func, axis=1)
        
        # Sort stations by distance
        stations = stations.sort_values(by="r")

        # Create a new stream with the sorted traces
        stream = Stream(self.traces)
        info = stream._get_common_channels_info()
        
        # Extract true station names
        true_stations = list(zip(*list(info.keys())))[1]

        sorted_traces = []
        # Sort traces according to the sorted station list
        for _, row in stations.iterrows():
            station = row.station
            if station in true_stations:
                st = stream.select(station=station)
                for tr in st:
                    sorted_traces.append(tr)
        
        self.traces = sorted_traces
        return self


def plot_traces(stream, picks_dict, 
                color_authors,
                out=None, show=True, 
                figsize=(12, 12), fontsize=10):
    """
    Plot the traces in a stream with phase pick annotations.

    Parameters:
        stream (Stream): ObsPy Stream containing the traces to be plotted.
        picks_dict (dict): Dictionary of picks with phase and author information.
                            keys (str):author name. 
                            value (dict): PIcks Oobject with the next data:
                                columns: network,station,arrival_time
        color_authors (dict): Dictionary.   keys (str):author name. 
                                            value (dict): dictionary.
                                                keys (str): phase name
                                                value (str): color name
        out (str, optional): File path to save the figure. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to False.
        figsize (tuple, optional): Figure size. Defaults to (12, 12).
        fontsize (int, optional): Font size for text in the plot. Defaults to 10.
        
    Returns:
        fig, ax: The created matplotlib figure and axes.
    """
    # Create subplots, one for each trace in the stream
    fig, ax = plt.subplots(len(stream), sharex=True, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=figsize)
    
    all_labels = []
    
    # Iterate through each trace and plot the data
    for i, tr in enumerate(stream):
        start = tr.stats.starttime.datetime
        end = tr.stats.endtime.datetime
        deltadt = (end - start).total_seconds()
        delta = tr.stats.delta
        
        # Generate time axis for the trace
        x = [start + dt.timedelta(seconds=x) for x in np.arange(0, deltadt + delta, delta)]
        
        # Plot the trace data
        ax[i].plot(x, tr.data, 'k')
        
        # Iterate through each author and plot picks if available
        for author, picks in picks_dict.items():
            ps_picks = picks[(picks["arrival_time"] >= start) & (picks["arrival_time"] <= end)]
            ps_picks = ps_picks[ps_picks["station"] == tr.stats.station]
            
            if not ps_picks.empty:
                for _, pick in ps_picks.iterrows():
                    pick_time = pick["arrival_time"]
                    ymin, ymax = ax[i].get_ylim()
                    phasehint = pick["phase_hint"]
                    label = f"{phasehint}-{author.capitalize()}"
                    station_pick = pick["station"]
                    
                    print(station_pick,phasehint,pick_time)
                    
                    # Avoid duplicate labels in the legend
                    if label not in all_labels:
                        phase_label = label
                        all_labels.append(label)
                    else:
                        phase_label = None
                    
                    # Plot vertical lines at pick times
                    ax[i].vlines(pick_time, ymin, ymax, color=color_authors[author][phasehint],
                                 label=phase_label, linewidth=2)
        
        
        ax[i].set_yticklabels([])  # Hide y-axis labels
        ax[i].text(0.05, 0.95, f'{tr.id}', transform=ax[i].transAxes, fontsize=fontsize,
                   style='italic', color="red", bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})
        
        # Set x-axis label
        ax[i].set_xlabel(f"Time", fontsize=14)
    
    # Add legend to the plot
    fig.legend(loc="upper right", ncol=4, bbox_to_anchor=(1, 1))
    fig.autofmt_xdate()
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure if output path is provided
    if out is not None:
        if not os.path.isdir(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
        fig.savefig(out)
    
    # Show the plot if required
    if show:
        plt.show()

    return fig, ax

def merge_stream(stream):
    """
    Merge traces in the stream. If traces with the same ID have different sampling rates,
    skip merging those traces.
    
    Parameters:
    stream (Stream): ObsPy Stream object with traces to be merged.

    Returns:
    Stream: Merged stream, with traces that have differing sampling rates left unmerged.
    """
    merged_stream = Stream()  # Initialize an empty stream for merged traces
    
    # Group traces by their id (same network, station, location, channel)
    for trace_id in set(tr.id for tr in stream):
        # Select all traces with the same id
        traces = stream.select(id=trace_id)
        
        # Check if all traces have the same sampling rate
        sampling_rates = set(tr.stats.sampling_rate for tr in traces)
        
        if len(sampling_rates) == 1:  # If all traces have the same sampling rate, merge them
            merged_traces = traces.merge()
            merged_stream += merged_traces
        else:
            print(f"Skipping merge for {trace_id}: differing sampling rates")
            merged_stream += traces  # Add the unmerged traces to the final stream

    return merged_stream

class Tracer():
    def __init__(self,url,mulpicks,stations,
                 preferred_author=None):
        self.url = url
        self.client = Client(url)
        self.mulpicks = mulpicks
        self.stations = stations
        self.preferred_author = preferred_author
    
    def _get_stream(self,starttime,endtime,
                    network_list=None,
                    remove_stations=None):
        
        all_st = Stream()
        
        self._mulpicks = self.mulpicks.copy()
        self._mulpicks.filter("arrival_time",start=starttime,end=endtime)
        station_ids = self._mulpicks.get_station_ids(self.preferred_author)
        
        # Initialize an empty dictionary to store the grouped items
        grouped_stations = {}
        # Iterate over the list of station names
        for station in station_ids:
            # Split the station name by '.' to get the first two characters (network code) and station ID
            network_code, station_id = station.split('.')
            
            if remove_stations is not None:
                if station_id in remove_stations:
                    # print("NO",network_code)
                    continue
            
            if network_list is not None:
                if network_code not in network_list:
                    # print("NO",network_code)
                    continue
            
            # Add the station ID (YYYY part) to the appropriate group in the dictionary
            if network_code not in grouped_stations:
                grouped_stations[network_code] = []
            
            grouped_stations[network_code].append(station_id)
        # print(grouped_stations)
        
        for network,stations in grouped_stations.items():
            station = ",".join(stations)
            print(network,station)
            # print(self.client)
            try:
                st = self.client.get_waveforms(network=network,
                                        station=station,
                                        location=",00,10",
                                        channel="*Z",
                                        starttime=UTCDateTime(starttime),
                                        endtime=UTCDateTime(endtime))
            except Exception as e:
                print(e)
                st = Stream()
                
            all_st += st
            
        return all_st
    
    def plot(self,starttime,endtime,
             network_list=None,
             remove_stations = None,
             sort_from_source=None,
             sort_by_first_arrival=None,
             **kwargs):
        
        st = self._get_stream(starttime,endtime,network_list=network_list,
                              remove_stations=remove_stations)
        st = merge_stream(st)
        
        stations_data = self.stations.data.copy()
        myst = MyStream(st.traces,stations_data)
        
        
        if sort_from_source is not None:
            source = (sort_from_source.longitude,
                      sort_from_source.latitude)
            myst.sort_from_source(source=source)
            myst.detrend().normalize()
        
        if sort_by_first_arrival:
            station,station_time = self._mulpicks.get_lead_station()
            
            station_info = stations_data[stations_data['station'] == station][['latitude', 'longitude']]
            lat, lon = station_info.iloc[0]
            source = (lon,lat)
            myst.sort_from_source(source=source)
            myst.detrend().normalize()
        
        picks_dict = {}
        colors_dict = {}
        for picks in self._mulpicks:
            picks_dict[picks.author] = picks.data
            colors_dict[picks.author] = {"P":picks.p_color,
                                         "S":picks.s_color,
                                         }
        fig,ax = plot_traces(myst,picks_dict=picks_dict,
                    color_authors=colors_dict,
                    **kwargs)
        
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
    st = r"/home/emmanuel/ecastillo/dev/delaware/monitor/downloads"
    
    picks = "/home/emmanuel/ecastillo/dev/delaware/monitor/picks/eqt/seismonitor_picks.csv"
    picks = pd.read_csv(picks,parse_dates=["arrival_time"])
    
    syn_picks = "/home/emmanuel/ecastillo/dev/delaware/monitor/picks/pykonal/single_eq.csv"
    syn_picks = pd.read_csv(syn_picks)
    origin_time = dt.datetime.strptime("2022-11-16 21:32:44.481999","%Y-%m-%d %H:%M:%S.%f")
    syn_picks["arrival_time"] = syn_picks["travel_time"].apply(lambda x: dt.timedelta(seconds=x)+origin_time)
    # print(syn_picks)
    # exit()
    
    
    syn2_picks = "/home/emmanuel/ecastillo/dev/delaware/monitor/picks/pykonal/single_eq_2.5.csv"
    syn2_picks = pd.read_csv(syn2_picks)
    origin_time = dt.datetime.strptime("2022-11-16 21:32:44.481999","%Y-%m-%d %H:%M:%S.%f")
    syn2_picks["arrival_time"] = syn2_picks["travel_time"].apply(lambda x: dt.timedelta(seconds=x)+origin_time)
    # print(syn_picks)
    
    st = read_st(st,leaving_gaps=False,files_format=".mseed")
    print(st)
    st = st.select(component="Z")
    # st = st.select(component="N",station="CA02A")
    st = st.trim(starttime=UTCDateTime("2022-11-16T21:32:30.000000Z"),
                    endtime=UTCDateTime("2022-11-16T21:33:30.000000Z"),)
    # st.plot(method="full")
    # st = st.merge(method=1)
    # print(st)
    # for tr in st:
    #     if isinstance(tr.data, np.ma.masked_array):
    #         tr.data = tr.data.filled()
    # # # st = st.split()

    
    myst = MyStream(st.traces,stations)
    myst.sort_from_source(source=(-103.99,31.63))
    myst.detrend().normalize()
    # myst.normalize()
    # plot_traces(myst,picks)
    
    # Define color map for different authors and phases
    color_authors = {
        "eqt": {"P": "blue", "S": "red"},
        "pykonal": {"P": "cyan", "S": "magenta"},
        "pykonal_2.5km": {"P": "green", "S": "purple"}
    }
    
    picks_list = {"eqt":picks,"pykonal":syn_picks,
                  "pykonal_2.5km":syn2_picks,}
    
    plot_traces(myst,picks_list)
    plt.show()
