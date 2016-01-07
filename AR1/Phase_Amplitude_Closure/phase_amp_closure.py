# coding: utf-8
import katdal
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.backends.backend_pdf import PdfPages
from katsdpscripts.RTS import git_info,get_git_path

import optparse

def phase_combinations(ant_list,look_up):
    l1,l2,l3 = [],[],[]
    for a1,a2,a3 in itertools.combinations(set( ant_list), 3):
        l1.append(look_up[a1,a2])
        l2.append(look_up[a1,a3])
        l3.append(look_up[a2,a3])
return a1,a2,a3

def amp_combinations(ant_list,look_up):
    l1,l2,l3,l4 = [],[],[],[]
    for a1,a2,a3,a4 in itertools.combinations(set( ant_list), 4):
        l1.append(look_up[a1,a2])
        l2.append(look_up[a1,a3])
        l3.append(look_up[a2,a4])
        l4.append(look_up[a3,a4])
    return a1,a2,a3,a4

def anglemean(th,axis=None):
    """ Return the mean of angles
    Multiply angles by 2 for an directionless orentation
    eg. polorisation """
    sa = np.nansum(np.sin(th),axis=axis)
    ca = np.nansum(np.cos(th),axis=axis)
    return np.arctan2(sa,ca)

def plot_phase_freq(channel_freqs,a123,title=''):
    """
    channel_freqs is an array of channel frequencys in Hz
    a123 is the closure quantity in radians 
    """
    fig = plt.figure(figsize=(20,10))
    plt.title(title)
    plot(channel_freqs/1e6,np.degrees(a123) )
    plt.ylim(-5,5)
    plt.grid(True)
    plt.ylabel('Mean Phase Closure angle(degrees)')
    plt.xlabel('Frequency (MHz)')
    plt.figtext(0.89, 0.11,git_info(get_git_path()), horizontalalignment='right',fontsize=10)
    return fig 

def plot_amp_freq(channel_freqs,a1234,title=''):
    """
    channel_freqs is an array of channel frequencys in Hz
    a1234 is the closure quantity  
    """
    fig = plt.figure(figsize=(20,10))
    plt.title(title)
    plot(channel_freqs/1e6,a1234 )
    plt.grid(True)
    plt.ylabel('Mean Amplitude Closure ')
    plt.xlabel('Frequency (MHz)')
    plt.figtext(0.89, 0.11,git_info(get_git_path()), horizontalalignment='right',fontsize=10)
    return fig 

# Parse command-line options and arguments
parser = optparse.OptionParser(usage='%prog [options] <data file>',
                               description='This script reduces a data file to produce a plots of the closure quantitiys in a pdf file.')
parser.add_option("-f", "--freq-chans", default=None,
                  help="Range of frequency channels to keep (zero-based, specified as 'start,end', default= %default)")
parser.add_option("-d", "--print_description", action="store_true",default=False,
                  help="Add an additional page that discribes the therory of the plots, default= %default)")

(opts, args) = parser.parse_args()


if len(args) < 1:
    raise RuntimeError('Please specify the data file to reduce')



nice_filename =  args[0].split('/')[-1]+ '_closure'
pp =PdfPages(nice_filename+'.pdf')

h5 = katdal.open('/data/sean/1451995933.h5')
h5.select()
for scan in h5.scans() :
    for pol in ['h','v'] :    
        h5.select(pol=pol)
        #h5.select(scans='track',targets='PKS1934-638')
        N_ants = len(h5.ants)
        antA = [h5.inputs.index(inpA) for inpA, inpB in h5.corr_products]
        antB = [h5.inputs.index(inpB) for inpA, inpB in h5.corr_products]

        full_vis = (np.concatenate((h5.vis[:], (h5.vis[:]).conj()), axis=-1))
        full_antA = np.r_[antA, antB]
        full_antB = np.r_[antB, antA]
        corrprods = zip(full_antA,full_antB)
        up = {}
        for i,(x,y)  in enumerate(zip(full_antA,full_antB)): # make lookup table
            up[x,y]=i
            up[y,x]=i
        title = "%s : pol %s  , target=%s , %s "%(args[0].split('/')[-1],pol,scan[2].name,scan[1])
        a1,a2,a3 = phase_combinations(full_antA,up,title)
        a123 =  anglemean(np.rollaxis(np.angle(full_vis[:,:,l1])-np.angle(full_vis[:,:,l2]) +np.angle(full_vis[:,:,l3]),0,2  ).reshape(full_vis.shape[1],-1)  ,axis=1 ) 
        fig = plot_phase_freq(h5.channel_freqs,a123)
        fig.savefig(pp,format='pdf')
        plt.close(fig)
 
        a1,a2,a3,a4 = amp_combinations(full_antA,up,title)
        a1234 =  np.nanmean(np.rollaxis((np.abs(full_vis[:,:,l1])*np.abs(full_vis[:,:,l4]))/(np.abs(full_vis[:,:,l2])*np.abs(full_vis[:,:,l3] ) ) ,0,2).reshape(full_vis.shape[1],-1),axis=-1) 
        fig = plot_amp_freq(h5.channel_freqs,a1234)
        fig.savefig(pp,format='pdf')
        plt.close(fig)


if opts.print_description :
    text = r''
    text +=r"""
     The Phase relationship equation is:
 
     $ \Phi_{12}  = \phi_{1} - \phi_{2} +  \phi_{12}  $
 
     $ \Phi_{13}  = \phi_{1} - \phi_{3} +  \phi_{13}  $
 
     $ \Phi_{23}  = \phi_{2} - \phi_{3} +  \phi_{23}  $
 
     $ \Phi_{12} - \Phi_{13} + \Phi_{23} =  \phi_{12} - \phi_{13} +\phi_{23} $
 
     For a point source , $\phi_{12} = \phi_{13} = \phi_{23} = 0 $
    """
    text +=r"""
     \n\n
     The Amplitude relationship equation is:
 
     $ A_{12}  = a_{1}a_{2}^{*}a_{12}S_{12}  $
 
     $ A_{13}  = a_{1}a_{3}^{*}a_{13}S_{13}  $
 
     $ A_{24}  = a_{2}a_{4}^{*}a_{24}S_{24}  $
 
     $ A_{34}  = a_{3}a_{4}^{*}a_{34}S_{34}  $  
 
     $ \frac{A_{12}A_{34}}{A_{13}A_{24}} =  \frac{a_{12}a_{34}}{a_{13}a_{24}} S $
 
     For a point source , $S_{12} = S_{13} = S_{24} = S_{34} = S $
    """
    plt.figtext(0.1,0.1,text,fontsize=10)
    fig.savefig(pp,format='pdf')
    plt.close(fig)
pp.close()
plt.close('all')
