from obspy import UTCDateTime,read,Stream
import pandas as pd
from read import read_st
import datetime as dt
from delaware.eqviewer import Station


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