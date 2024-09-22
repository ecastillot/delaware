from obspy import read
from obspy import UTCDateTime
from obspy.core.stream import Stream
import glob
import os
import concurrent.futures as cf
import numpy as np
import copy
def merge_leaving_gaps(st, start_time, end_time, debug=False):
    """
    This function will merge an obspy stream but will leave gaps between traces if they are larger than one sample
    :param st: the obspy stream to be merged
    :param start_time: start time of the window of interest (datetime object)
    :param end_time: end time of the window of interest (datetime object)
    :param debug: if True, turn on debugging mode with more print outs
    :return:
    """
    if debug:
        print('Merging leaving gaps')
    # trying to write a code that will deal with merging together streams without creating masked arrays
    st_new = Stream()

    # pulling out the original sampling rate
    orig_sampling_rate = copy.deepcopy(st[0].stats.sampling_rate)

    # getting the station and channels strings
    stat_chans = []
    for tr in st:
        if tr.stats.station + '.' + tr.stats.channel not in stat_chans:
            stat_chans.append(tr.stats.station + '.' + tr.stats.channel)

    # now looping through each of them
    for scn in range(len(stat_chans)):
        temp_stat = stat_chans[scn].split('.')[0]
        temp_chan = stat_chans[scn].split('.')[1]
        if debug:
            print(temp_stat)
            print(temp_chan)
        # pulling out all of the data for that station
        st2 = st.copy().select(station=temp_stat, channel=temp_chan).sort()

        # now looping through each trace of that station
        st_new2 = Stream()
        for trn in range(len(st2)):
            # resampling the stream if it has a different sampling rate to the first trace
            if st2[trn].stats.sampling_rate != orig_sampling_rate:
                if debug:
                    print(trn)
                    print(orig_sampling_rate)
                    print(st2[trn].stats.sampling_rate)
                    print(st2[trn])
                st2[trn].resample(orig_sampling_rate)
                st2[trn].data = np.array(st2[trn].data, dtype=np.int32)
                if debug:
                    print(st2[trn])

            # we want to add it to the empty stream, perform the merge, and then output
            if len(st_new2) == 0:
                st_new2 += st2[trn]
            else:
                # check if the start time is before the end time of the trace in the empty stream
                if st2[trn].stats.starttime <= st_new2[0].stats.endtime:
                    # if it is then add it to the stream and merge
                    st_new2 += st2[trn]
                    st_new2.merge(method=1)
                # if it is not then output the trace to the original empty stream and then make
                # st_new2 empty again so we start building a new trace
                # increased this to 2.5 because if one sample is missing between 2 traces then it would be a
                # 2 sample wide gap
                elif st2[trn].stats.starttime - st_new2[0].stats.endtime <= st2[trn].stats.delta * 2.5:
                    # if there is just one sample missing then merge it in with a interpolated fill value
                    st_new2 += st2[trn]
                    st_new2.merge(method=1, fill_value='interpolate')
                else:
                    st_new += st_new2[0]
                    st_new2 = Stream()
                    st_new2 += st2[trn]

        # merging the final trace
        st_new += st_new2[0]

    # trimming the traces to make sure they don't go into the next day
    start_time = UTCDateTime(start_time)
    end_time = UTCDateTime(end_time)

    if debug:
        print(start_time)
        print(end_time)
        print(st_new)

    # looping through each of the traces and trimming them if appropriate
    for tr in st_new:
        if debug:
            print(tr)
        # minusing a sample off the end time so we don't get any overlap
        nend_time = end_time - tr.stats.delta
        if tr.stats.starttime > end_time or tr.stats.endtime < start_time:
            st_new.remove(tr)
        elif tr.stats.starttime < start_time and tr.stats.endtime > nend_time:
            tr.trim(start_time, nend_time)
        elif tr.stats.starttime < start_time and tr.stats.endtime <= nend_time:
            tr.trim(start_time, tr.stats.endtime)
        elif tr.stats.starttime >= start_time and tr.stats.endtime > nend_time:
            tr.trim(tr.stats.starttime, nend_time)
        else:
            pass

    # returns the stream of merged traces
    return st_new

def read_st(folder,out=None,plot=False,digitizer=None,leaving_gaps=True,files_format=".seed",debug=False):
    
    st = Stream()
    for dp, dn, filenames in os.walk(folder):
            for f in filenames:
                filename, file_extension = os.path.splitext(f)
                if file_extension == files_format:
                    if digitizer != None:
                        if digitizer not in filename:
                            continue
                    seed_path = os.path.join(dp, f)
                    
                    if files_format == ".hsf":
                        str = hsf_to_obspy(seed_path)
                    elif files_format == ".seed":
                        str = read(seed_path)
                    elif files_format == ".mseed":
                        str = read(seed_path)
                    else:
                        continue
                    st+= str

                    if debug:
                        print(seed_path)
    if leaving_gaps:
        starttime = st[0].stats.starttime
        endtime = st[-1].stats.endtime
        st = merge_leaving_gaps(st,starttime,endtime,debug=debug)
    else:
        st = st.merge(method=1,fill_value=None)
    # for tr in st:
    #     if isinstance(tr.data, np.ma.masked_array):
    #         tr.data = tr.data.filled()
    # st.plot()
    if out != None:
        if not os.path.isdir(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))

        st.write(out,format="MSEED")
    if plot:
        st.plot()
    return st


if __name__ == "__main__":
    # # import matplotlib.pyplot as plt
    # # folderpath = r"\\esg.net\datashare\ISM\Ecopetrol\ECP_DataStorage\stations\2023-02-23"
    # # outpath = r"C:\Users\Emmanuel.Castillo\OneDrive - ESG Solutions\ESG\ecastillot\Dev\Read_ESG_stdata\out"
    # # out = os.path.join(folderpath,"out","test.mseed")
    # # # st = read_st(folderpath,out)
    # # st = read(out)
    # # st = st.select(channel = "*Z")
    # # st = st.trim(starttime=UTCDateTime(2023,2,23,0,0),endtime=UTCDateTime(2023,2,23,6,0))
    # # # st = st.detrend()
    # # st = st.normalize()
    # # st.plot(outfile=r"C:\Users\Emmanuel.Castillo\OneDrive - ESG Solutions\ESG\ecastillot\Dev\Read_ESG_stdata\test.png")
    # # # plt.show()

    import matplotlib.pyplot as plt
    # folderpath = r"\\esg.net\datashare\ISM\Ecopetrol\ECP_FieldTest\AP06A_2023_02_16_26\PPSD_RealStation\stations"
    # folderpath = r"C:\Users\Emmanuel.Castillo\OneDrive - ESG Solutions\ESG\ecastillot\Dev\Read_ESG_stdata\test"


    # folderpath = r"\\esg.net\datashare\ISM\Ecopetrol\ECP_FieldTest\AP06A_2023_02_16_26\PPSD_input"
    # out = os.path.join(os.path.dirname(folderpath),"st","AP06A_FEB2023_test.mseed")

    folderpath = r"\\esg.net\datashare\ISM\Ecopetrol\ECP_questions\AllNetworks_2023_03_10_09_10\CA_CH"
    out = os.path.join(os.path.dirname(folderpath),"st_CA_CH","CA_CH_2023_03_10_09_10.mseed")


    # st = read_st(folderpath)
    # st = read_st(folderpath,out=out)
    st = read_st(folderpath,out=out)
    # st = read(out)
    print(st)
    st = st.select(channel = "*Z")
    # # st = st.trim(starttime=UTCDateTime(2023,2,23,0,0),endtime=UTCDateTime(2023,2,23,6,0))
    # # st = st.detrend()
    # # st = st.normalize()
    st.plot(show=True,method="full",automerge=True)
    plt.tight_layout()
    # plt.show()

