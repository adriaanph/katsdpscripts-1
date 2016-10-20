#!/usr/bin/python
# Script that analyzes fast gain stability measurement at SCP
#

import numpy as np
import katdal
import matplotlib.pyplot as plt 
# figure, plot, psd, imshow, colorbar, legend, xlabel, ylabel, subplot, ylim, title, suptitle


def sliding_rms(x, win_length):
    """ @return: RMS computed over sliding windows on x, zero-padded to align with x. """
    W = np.min([len(x), int(win_length+0.1)])
    rms1 = lambda x_block: (np.sum((x_block-np.average(x_block))**2)/float(W))**.5
    results = list(0*x) # Allocate & fill with zeros
    for i in range(len(x)-W): # Process sliding blocks of length N
        results[W//2+i] = rms1(x[i:i+W])
    return np.asarray(results)

def fit_avg(x, win_length):
    """ @return: block average over x."""
    W = np.min([len(x), int(win_length+0.1)])
    results = list(x) # Copy - this ensures padding with original values for last ~window
    for i in range(0,len(x)-W+1,W-1): # W-1 steps means the last preceding point is the first next point 
        results[i:i+W] = np.ones(W)*np.average(x[i:i+W])
    return np.asarray(results)

def analyze(h5, ant, t_spike_start, t_spike_end):
    """
        @param t_spike_start, t_spike_end: start & end times of noisy time series to be excluded from analysis, in [sec]
    """
    # Select the correct scan. It's always the last 'track'
    h5.select(reset="TFB", scans='track')
    h5.select(scans=h5.scan_indices[-1], ants=[ant])

    t = h5.timestamps-h5.timestamps[0]
    dt = h5.dump_period
    if (abs(dt/(t[1]-t[0])-1) > 0.1):
        print("Discrepancy between dump period and recorded time intervals, using latter")
        dt = np.mean(np.diff(h5.timestamps))
    rate = 1./dt
    pols = [i for i,x in enumerate(h5.corr_products) if x[0][-1]==x[1][-1]] # Only XX & YY
    p_h = np.abs(h5.vis[:,:,pols[0]].squeeze())
    p_v = np.abs(h5.vis[:,:,pols[1]].squeeze())
    p_hv = np.abs(h5.vis[:,:,[i for i,x in enumerate(h5.corr_products) if x[0][-1]!=x[1][-1]][0]].squeeze())

    # Plots to identify RFI-free bits of spectrum
    plt.figure(figsize=(16,8));
    plt.suptitle("%s: %s"%(h5.file.filename, ant))

    plt.subplot(2,1,1) # H & V sigma/mu spectra
    plt.plot(h5.channels, np.std(p_h,axis=0)/np.mean(p_h,axis=0))
    plt.plot(h5.channels, np.std(p_v,axis=0)/np.mean(p_v,axis=0))
    K = 1/np.sqrt(h5.channel_width*dt) # Expected radiometer scatter
    plt.ylim(K/2.,3*K)
    plt.ylabel(r"$\sigma/\mu$ []")
    plt.title("Complete spectrum")

    plt.subplot(2,1,2) # HV power
    plt.plot(h5.channels, 10*np.log10(np.mean(p_hv,axis=0)))
    plt.ylabel(r"HV [dB]")

    plt.xlabel("Frequency [channel #]");

    # Identify pristine chunks of spectrum e.g. from the above
    M = int(10/dt) # Minimum = 20/dt MHz to beat system noise
    ch_chunks = [range(100+M*n,100+M*(n+1)) for n in range(1,(len(h5.channels)-200)/M)] # Omit 100 channels at both edges

    snr_h = [np.mean(np.std(p_h[:,C],axis=0)/np.mean(p_h[:,C],axis=0)) for C in ch_chunks]
    snr_v = [np.mean(np.std(p_v[:,C],axis=0)/np.mean(p_v[:,C],axis=0)) for C in ch_chunks]
    snr = 1/2**.5 * np.sqrt(np.asarray(snr_h)**2+np.asarray(snr_v)**2) # Average over frequency

    snr_flags = snr<1.05*K # at most 5% more than expected
    print(snr[snr_flags])
    
    hv = np.asarray([np.std(p_hv[:,C]/np.std(p_hv[:,C],axis=0))-1 for C in ch_chunks])
    print(hv[snr_flags])
    hv_flags = hv<10*np.percentile(hv,10) # grossly unstable compared to the typical best

    ch_chunks = np.asarray(ch_chunks)[snr_flags*hv_flags]
    print(ch_chunks)
    
    plt.figure(figsize=(16,8))
    plt.suptitle("%s: %s"%(h5.file.filename, ant))
    plt.subplot(2,1,1) # H & V sigma/mu spectra
    for ch in ch_chunks:
        plt.plot(h5.channels[ch], np.std(p_h[:,ch],axis=0)/np.mean(p_h[:,ch],axis=0))
        plt.plot(h5.channels[ch], np.std(p_v[:,ch],axis=0)/np.mean(p_v[:,ch],axis=0))
    plt.plot(h5.channels,K+0*h5.channels,'k,')
    plt.ylim(K/2.,3*K)
    plt.ylabel(r"$\sigma/\mu$ []")
    plt.title("Pristine spectrum")

    plt.subplot(2,1,2) # HV power
    for ch in ch_chunks:
        plt.plot(h5.channels[ch], 10*np.log10(np.mean(p_hv[:,ch],axis=0)))
    plt.plot(h5.channels,K+0*h5.channels,'k,');
    plt.ylabel(r"HV [dB]")

    plt.xlabel("Frequency [channel #]"); 

    # Time series H & V
    plt.figure(figsize=(16,8))
    plt.suptitle("%s: %s"%(h5.file.filename, ant))
    for ch in ch_chunks:
        plt.subplot(2,1,1)
        plt.plot(t, np.mean(p_h[:,ch],axis=1)/np.mean(p_h[:,ch]))
    plt.xlabel("time [sec]")
    plt.ylabel(r"$\delta P/P$ [linear]")

    for ch in ch_chunks:
        plt.subplot(2,1,2)
        plt.plot(t, np.mean(p_v[:,ch],axis=1)/np.mean(p_v[:,ch]))
    plt.xlabel("time [sec]")
    plt.ylabel(r"$\delta P/P$ [linear]");
    
    # Generate the flags for spikes in time series
    t_A, t_B = None, None # Flags identifying the clean & spike windows respectively
    if t_spike_start is not None and t_spike_end is not None:
        AB = t_spike_end-t_spike_start
        if (t_spike_end < t[len(t)/2]): # Spike in first half of data, choose clean window after spike
            t_A = np.nonzero(np.abs(t-(t_spike_end+AB/2.))<=AB/2.) # Clean
            t_B = np.nonzero(np.abs(t-(t_spike_end-AB/2.))<=AB/2.) # Spike here
        else: # Spike in second half of data, choose clean window before spike
            t_A = np.nonzero(np.abs(t-(t_spike_start-AB/2.))<=AB/2.) # Clean
            t_B = np.nonzero(np.abs(t-(t_spike_start+AB/2.))<=AB/2.) # Spike here

    # Debug in case time-domain spikes are noticed
    if t_A is not None and t_B is not None:
        # Time domain
        plt.figure(figsize=(16,8))
        plt.suptitle("%s: %s"%(h5.file.filename, ant))
        for ch in ch_chunks:
            plt.plot(t[t_A], (np.mean(p_h[:,ch],axis=1)/np.mean(p_h[:,ch]))[t_A])
            plt.plot(t[t_A], (np.mean(p_v[:,ch],axis=1)/np.mean(p_v[:,ch]))[t_A])
            plt.plot(t[t_B], (np.mean(p_h[:,ch],axis=1)/np.mean(p_h[:,ch]))[t_B])
            plt.plot(t[t_B], (np.mean(p_v[:,ch],axis=1)/np.mean(p_v[:,ch]))[t_B])
        plt.xlabel("time [sec]")
        plt.ylabel(r"$\delta P/P$ []")

        # Spectral domain
        plt.figure(figsize=(16,8))
        plt.suptitle("%s: %s"%(h5.file.filename, ant))
        plt.plot(h5.channels, 10*np.log10(np.mean(p_h[t_B,:].squeeze(),axis=0) / np.mean(p_h[t_A,:].squeeze(),axis=0)))
        plt.plot(h5.channels, 10*np.log10(np.mean(p_v[t_B,:].squeeze(),axis=0) / np.mean(p_v[t_A,:].squeeze(),axis=0)))
        plt.ylim(-0.1,0.1)
        plt.xlabel("Frequency [channel #]")
        plt.ylabel("P(spike)/P(nospike) [dB#]")
  
    # Only use the data from start up to just before the spike
    if t_A is not None and t_B is not None:
        if (np.min(t_A)<np.min(t_B)): # Spike is after clean data
            t_Z = np.min(t_B)
            t = t[:t_Z]
            p_h = p_h[:t_Z,:]
            p_v = p_v[:t_Z,:]
        else: # Spike precedes clean data
            t_Z = np.min(t_A)
            t = t[t_Z:]
            p_h = p_h[t_Z:,:]
            p_v = p_v[t_Z:,:]

    # PSD of individual channel chunks
    plt.figure(figsize=(12,8))
    plt.suptitle("%s: %s"%(h5.file.filename, ant))
    for ch in ch_chunks:
        plt.subplot(2,1,1)
        plt.psd(np.mean(p_h[:,ch],axis=1)-np.mean(p_h[:,ch]), Fs=1/dt, NFFT=len(t))

    for ch in ch_chunks:
        plt.subplot(2,1,2)
        plt.psd(np.mean(p_v[:,ch],axis=1)-np.mean(p_v[:,ch]), Fs=1/dt, NFFT=len(t))

    # All good channels combined
    plt.figure(figsize=(12,6))
    plt.suptitle("%s: %s"%(h5.file.filename, ant))
    P_h=np.take(p_h,ch_chunks,axis=1).reshape(len(t),np.prod(ch_chunks.shape))
    plt.psd(np.mean(P_h,axis=1)-np.mean(P_h), Fs=1/dt, NFFT=len(t))
    P_v=np.take(p_v,ch_chunks,axis=1).reshape(len(t),np.prod(ch_chunks.shape))
    plt.psd(np.mean(P_v,axis=1)-np.mean(P_v), Fs=1/dt, NFFT=len(t));       

    # Measurements in each identified good channel chunk individually
    plt.figure(figsize=(12,20))
    plt.suptitle("%s: %s"%(h5.file.filename, ant))
    for i,(pol,p_t) in enumerate([("H",p_h), ("V",p_v)]):
        for ch in ch_chunks:
            p_tch = np.mean(p_t[:,ch],axis=1)/np.mean(p_t[:,ch])
            plt.subplot(4,1,2*i+1)
            plt.plot(t, p_tch,'+', t, fit_avg(p_tch,5/dt), '.')
            plt.ylabel("Sampled power [linear]")
            plt.title("%s:%s %s pol @ %.f Hz"%(ant,h5.receivers[ant],pol,rate))
            plt.subplot(4,1,2*i+2)
            plt.plot(t, 100*sliding_rms(p_tch,5/dt), label="ch ~%.f"%ch.mean());  plt.ylabel("RMS over 5 sec [%]")
            plt.plot(t, 0.10+0*t, 'k--') # Spec limit
        plt.legend()
        plt.ylim(0,0.15)
    plt.xlabel("time [sec]");

    # Measurements combined over all identified good channel chunks
    plt.figure(figsize=(12,20))
    plt.suptitle("%s: %s"%(h5.file.filename, ant))
    for i,(pol,p_t) in enumerate([("H",P_h), ("V",P_v)]):
        p_t = np.mean(p_t,axis=1)/np.mean(p_t)
        plt.subplot(4,1,2*i+1)
        plt.plot(t, p_t,'+', t, fit_avg(p_t,5/dt), '.')
        plt.ylabel("Sampled power [linear]")
        result = 100*sliding_rms(p_t,5/dt)
        plt.title("%s:%s %s pol @ %.1f Hz: 95th pct %.3f%%"%(ant,h5.receivers[ant],pol,rate,np.percentile(result,95)))
        plt.subplot(4,1,2*i+2)
        plt.plot(t, result);  plt.ylabel("RMS over 5 sec [%]")
        plt.plot(t, 0.10+0*t, 'k--') # Spec limit
        plt.ylim(0,0.15)
    plt.xlabel("time [sec]");


import optparse
# Parse command-line opts and arguments
parser = optparse.OptionParser(usage="%prog [opts] <HDF5 file>",
                               description="This processes an HDF5 dataset and generates figures.")
parser.add_option("--ant", type='string', default=None,
                  help="Specific antenna to run analysis for (default = %default)")
parser.add_option("--spike-start", type='float', default=None,
                  help="Start of spike in time series to be omitted, in seconds (default = %default)")
parser.add_option("--spike-end", type='float', default=None,
                  help="End of spike in time series to be omitted, in seconds (default = %default)")

(opts, args) = parser.parse_args()
if len(args) != 1 or not args[0].endswith('.h5'):
    raise RuntimeError('Please specify a single HDF5 file as argument to the script')

filename = args[0]
ant = opts.ant
t_spike_start, t_spike_end = opts.spike_start, opts.spike_end

h5 = katdal.open(filename)
#print(h5)
#print(h5.receivers)
if ant:
    analyze(h5, ant, t_spike_start, t_spike_end)
else:
    for ant in h5.ants:
        analyze(h5, ant.name, t_spike_start, t_spike_end)
#plt.show()


