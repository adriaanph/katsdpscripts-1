#!/usr/bin/python
# Read data from a file and plot some stats and find pulses
import numpy as np
import optparse
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
from katsdpscripts import git_info

def MAD_median(data):
    """Median absolute deviation (MAD) is a robust measure of the variability."""
    import time
    start_0 = time.time()
    med = np.median(data)
    start_1 = time.time()
    mad = np.median(np.abs(data-med) )
    end_time = time.time()
    print("Time for operations , 1st median = %f3 2nd median = %f3 "%(start_1-start_0,end_time-start_1))
    return mad,med

def join_pulses(data,pulse_gap=10):
    """Find all the timestamps with the pulse"""
    pulse_listtmp = []
    pulse_list = []
    temp = []
    for x in xrange(1,data.shape[0]):
        if data[x] -data[x-1] > pulse_gap : # new pulse
            pulse_listtmp.append(temp)
            temp = []
            temp.append(data[x])
        else :
            temp.append(data[x])
    if len(temp) > 0 :
        pulse_listtmp.append(temp)
        temp = []
    for x in pulse_listtmp :
        pulse_list.append(np.array([np.min(x),np.max(x)]))  
    return np.array(pulse_list)

def map_to_raw_data(pulselist,avg_num=256,window_length=256,offset=0):
    """Find all the timestamps with the pulse"""
    #print pulselist
    output = np.zeros_like(pulselist)
    for i,(pmin,pmax) in enumerate(pulselist):
        pmax = pmax+window_length # rolling window forward
        pmin,pmax = pmin*avg_num,pmax*avg_num # Undo the average period
        output[i,:] = offset+pmin,offset+pmax # This is because of memory problems we break up the data
    return output

def rolling_window(a, window,axis=-1,pad=False,mode='reflect',**kargs):
    """
        This function produces a rolling window shaped data with the rolled data in the last col
        a      :  n-D array of data
        window : integer is the window size
        axis   : integer, axis to move the window over
        default is the last axis.
        pad    : {Boolean} Pad the array to the origanal size
        mode : {str, function} from the function numpy.pad
        One of the following string values or a user supplied function.
        'constant'      Pads with a constant value.
        'edge'          Pads with the edge values of array.
        'linear_ramp'   Pads with the linear ramp between end_value and the
        array edge value.
        'maximum'       Pads with the maximum value of all or part of the
        vector along each axis.
        'mean'          Pads with the mean value of all or part of the
        con  vector along each axis.
        'median'        Pads with the median value of all or part of the
        vector along each axis.
        'minimum'       Pads with the minimum value of all or part of the
        vector along each axis.
        'reflect'       Pads with the reflection of the vector mirrored on
        the first and last values of the vector along each
        axis.
        'symmetric'     Pads with the reflection of the vector mirrored
        along the edge of the array.
        'wrap'          Pads with the wrap of the vector along the axis.
        The first values are used to pad the end and the
        end values are used to pad the beginning.
        <function>      of the form padding_func(vector, iaxis_pad_width, iaxis, **kwargs)
        see numpy.pad notes
        **kargs are passed to the function numpy.pad
        
        Returns:
        an array with shape = np.array(a.shape+(window,))
        and the rolled data on the last axis
        
        Example:
        import numpy as np
        data = np.random.normal(loc=1,scale=np.sin(5*np.pi*np.arange(10000).astype(float)/10000.)+1.1, size=10000)
        stddata = rolling_window(data, 400).std(axis=-1)
        """
    if axis == -1 : axis = len(a.shape)-1
    if pad :
        pad_width = []
        for i in xrange(len(a.shape)):
            if i == axis:
                pad_width += [(window//2,window//2 -1 +np.mod(window,2))]
            else :
                pad_width += [(0,0)]
        a = np.pad(a,pad_width=pad_width,mode=mode,**kargs)
    a1 = np.swapaxes(a,axis,-1) # Move target axis to last axis in array
    shape = a1.shape[:-1] + (a1.shape[-1] - window + 1, window)
    strides = a1.strides + (a1.strides[-1],)
    return np.lib.stride_tricks.as_strided(a1, shape=shape, strides=strides).swapaxes(-2,axis) # Move original axis to


def real_time(file_position,sync_time,first_timestamp):
    ts=1.0/np.float128(1712e6)
    return np.float128(sync_time) + ts*(np.float128(timestamp_value)+np.float128(file_position) )

# Set up standard script options
parser = optparse.OptionParser(usage='%prog [options] <data file>',
                               description='This script produces some text plots so that data can be examind')
# Add experiment-specific options
#parser.add_option('-p', '--power-time', action="store_true", default=False,
#                  help='Plot a text power vs time graph (default=%default)')

parser.add_option('-w','--window',  type='int', default=256,
                  help='The size of the rolling windo for detection,  (default=%default)')
parser.add_option('-n','--num',  type='int', default=256,
                  help='Number of samples to average,  (default=%default)')
parser.add_option('-c','--chunk-size',  type='int', default=8192,
                  help='The size of the chunk to process as a factor of 32768,  (default=%default)')
parser.add_option('-s','--sync',  type='int', default=0,
                  help='The sync time epoch as a unix timestamp,  (default=%default)')
parser.add_option('-t','--first-timestamp',  type='int', default=0,
                  help='The firsts timestamp value   ,  (default=%default)')
parser.add_option('--plot', '--plotting', action="store_true", default=False,
                  help='Plot a graphs into a pdf file (default=%default)')

parser.add_option('-d','--detection',  type='float', default=8,
                  help='The detection level to use to find pulses in sigma(ish)  ,  (default=%default)')

#major/minor
# Set default value for any option (both standard and experiment-specific options)
#parser.set_defaults(description='UHF signal generator track',dump_rate=1.0,nd_params='off')
# Parse the command line
opts, args = parser.parse_args()


# Values to be read in from Params
sync_time =  opts.sync #Pulse100-16dB-Noise20dB-V.npy.epoch
timestamp_value = opts.first_timestamp # Pulse100-16dB-Noise20dB-V.npy.timestamp
ts=1.0/1712e6     # seconds per data point
avg_num = opts.num
window_length = opts.window
chunk_size = opts.chunk_size*32768
trans = slice(0,chunk_size) # First look plots 
plotting = opts.plot
if len(args) ==0 :
    raise RuntimeError('No file passed to the script')

data = np.load(args[0] , mmap_mode='r')

if plotting :   
    nice_filename =  args[0].split('/')[-1]+ '_Pulse_report'
    pp = PdfPages(nice_filename+'.pdf')
    fig = plt.figure()
    aaa = np.histogram(data[trans],bins=np.arange(2**6+1)-(2**5-.5) )
    plt.title('Histogram of first Chunk')
    plt.plot(aaa[1][1:]-0.5,(aaa[0]) )
    plt.xlim(-32,32)
    plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)
    fig.savefig(pp,format='pdf')
    plt.close(fig)

old_edge = 0
for new_edge in xrange(chunk_size,data.shape[0],chunk_size):
    trans = slice(old_edge,new_edge)
    old_edge = new_edge
    avg_data = (np.abs(data[trans]).reshape(-1,avg_num).mean(axis=-1)).astype(np.float)**2
    print avg_data.shape,window_length,trans
    rolled = rolling_window(avg_data, window=window_length)

    # choice of Measure  ?
    #measure = (rolled.mean(axis=-1))
    #measure = (rolled[...].std(axis=-1))
    measure = (rolled[...].std(axis=-1)/rolled.mean(axis=-1))
    mad,med =MAD_median(measure)
    #print mad,med, med+8*mad,med-8*mad

    if plotting and old_edge==chunk_size: # First chunk
        print "First Chunk"
        fig = plt.figure()
        plt.plot(avg_data)
        fig.savefig(pp,format='pdf')
        plt.close(fig)
        fig =plt.figure()
        plt.plot(measure)
        plt.hlines(med-opts.detection*mad,0,measure.shape[0])
        plt.hlines(med+opts.detection*mad,0,measure.shape[0],'r')
        plt.ylim(None,med+2.5*opts.detection*mad)
        plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)
        fig.savefig(pp,format='pdf')
        plt.close(fig)
        fig =plt.figure()
        plt.semilogy(np.abs(measure - med) / mad)
        plt.grid()
        plt.ylim(1,None)
        plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)
        fig.savefig(pp,format='pdf')
        plt.close(fig)

    pulse = (measure>med+opts.detection*mad) + (measure<med-opts.detection*mad)
    pulse_list = join_pulses(pulse.nonzero()[0])
    raw_data = map_to_raw_data(pulse_list,avg_num=avg_num,window_length=window_length,offset=trans.start)
    if plotting :
        for pmin,pmax in raw_data:
            selection = slice(pmin,pmax)
            plt.figure()
            plt.plot(1e6*ts*np.arange(data[selection].shape[0]),data[selection].astype(np.float)**2)
            a,b = plt.ylim()
            ptime = real_time((selection.start+selection.stop)/2.,sync_time=sync_time,first_timestamp=timestamp_value)
            plt.title("Pulse time is %33.22f seconds"%(ptime))
            plt.vlines(1e6*ts*data[selection].shape[0]/2.,a,b)    
            plt.figtext(0.89, 0.11,git_info(), horizontalalignment='right',fontsize=10)
            fig.savefig(pp,format='pdf')            
            plt.close(fig)
            break
    for pmin,pmax in raw_data:
        selection1 = slice(pmin,pmin+abs(pmax-pmin)//2)
        selection2 = slice(pmin+abs(pmin-pmax)//2,pmax)
        pchange = (data[selection2].astype(np.float)**2).mean() - (data[selection1].astype(np.float)**2).mean()
        ptime = real_time((pmin+pmax)/2.,sync_time=sync_time,first_timestamp=timestamp_value)
        up_down = 'up  '
        if np.signbit(pchange):
            up_down = 'down'
        print("Pulse power change %s %.2f db & time is %33.12f seconds"%(up_down,10*np.log10(np.abs(pchange)),ptime))
    
if plotting :
    pp.close()
    plt.close('all')






