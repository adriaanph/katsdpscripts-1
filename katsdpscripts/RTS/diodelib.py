#!/usr/bin/python
import katdal as katfile
import scape
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def read_and_plot_data(filename,output_dir='.',freq_band = 256e6):
    nice_filename =  filename.split('/')[-1].split('.')[0]+ '_T_sys_T_nd'
    pp = PdfPages(output_dir+'/'+nice_filename+'.pdf')

    h5 = katfile.open(filename)

    ants = h5.ants
    n_ants = len(ants)

    colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    pols = ['v','h']
    diode= 'coupler'
    fig1 = plt.figure(1,figsize = (15,5))
    fig2 = plt.figure(2,figsize = (15,5))
    rx_serial = str(4)
    rx_band = 'l'
    for pol in pols:
        for a,col in zip(ants,colour):    
            ant = a.name
            ant_num = int(ant[3])
            air_temp = np.mean(h5.sensor['Enviro/air_temperature'])
            diode_filename = '/var/kat/katconfig/user/noise-diode-models/mkat/rx.'+rx_band+'.'+rx_serial+'.'+pol+'.csv'
            diode_file = np.recfromcsv(diode_filename,names=['f','t_e_cal'])
            nd = scape.gaincal.NoiseDiodeModel(freq = diode_file['f']/1e6,temp = diode_file['t_e_cal'])
            
            s = h5.spectral_windows[0]
            f_c = s.centre_freq
            #cold data
            h5.select(ants=a.name,pol=pol,freqrange=(f_c - freq_band/2, f_c + freq_band/2),targets = 'OFF',scans='track')
            freq = h5.channel_freqs
            nd_temp = nd.temperature(freq / 1e6)
            cold_data = np.ma.array(h5.vis[:].real,mask=h5.flags()[:],fill_value=np.nan)
            on = h5.sensor['Antennas/'+ant+'/nd_coupler']
            buff = 1
            n_off = ~(np.roll(on,buff) | np.roll(on,-buff))
            n_on = np.roll(on,buff) & np.roll(on,-buff)
            cold_off = n_off
            cold_on = n_on
            #hot data
            h5.select(ants=a.name,pol=pol,freqrange=(f_c - freq_band/2, f_c + freq_band/2),targets = 'Moon',scans='track')
            hot_data = np.ma.array(h5.vis[:].real,mask=h5.flags()[:],fill_value=np.nan)
            on = h5.sensor['Antennas/'+ant+'/nd_coupler']
            buff = 1
            n_off = ~(np.roll(on,buff) | np.roll(on,-buff))
            n_on = np.roll(on,buff) & np.roll(on,-buff)
            hot_off = n_off
            hot_on = n_on
            cold_spec = np.median(cold_data[cold_off,:,0].filled(np.nan),0)
            hot_spec = np.median(hot_data[hot_off,:,0].filled(np.nan),0)
            cold_nd_spec = np.median(cold_data[cold_on,:,0].filled(np.nan),0)
            hot_nd_spec = np.median(hot_data[hot_on,:,0].filled(np.nan),0)
             
            hs,f = hot_spec,freq
            cs = cold_spec
            hns = hot_nd_spec
            cns = cold_nd_spec
            Y = hs / cs
            HPBW = 1.18 * (180/np.pi) *(3e8/(13.5*f))
            om = 1.133 * HPBW**2
            R = 0.25
            Thot = 225 * (np.pi * R**2)/om 
            Tsys = (Thot)/(Y-1)
            Ydiode = hns / cns
            Tdiode = (Thot + Tsys*(1-Ydiode))/(Ydiode-1)
            

            
            plt.figure(1)
            p = 1 if pol == 'v' else 2
            plt.subplot(n_ants,2,p)
            plt.ylim(1,40)
            if p ==ant_num * 2-1: plt.ylabel(ant)
            plt.plot(f,Tdiode,'b',label='Measurement')
            #outfile = file('%s/%s.%s.%s.csv' % (output_dir,ant, diode, pol.lower()), 'w')
            #outfile.write('#\n# Frequency [Hz], Temperature [K]\n')
            # Write CSV part of file
            #outfile.write(''.join(['%s, %s\n' % (entry[0], entry[1]) for entry in zip(f[((fs>1.2e9) & (fs < 1.95e9))],d[((fs>1.2e9) & (fs < 1.95e9))])]))
            #outfile.close()
            plt.plot(f,nd_temp,'k',label='Model')
            plt.grid()
            plt.legend()
            plt.figure(2)
            plt.subplot(n_ants,2,p)
            plt.ylim(10,60)
            if p == ant_num * 2 -1: plt.ylabel(ant)
            plt.plot(f,Tsys,'b')
            plt.grid()
        
    plt.figure(1)
    plt.subplot(n_ants,2,1)
    plt.title('Coupler Diode: H pol')
    plt.subplot(n_ants,2,2)
    plt.title('Coupler Diode: V pol')

    plt.figure(2)
    plt.subplot(n_ants,2,1)
    plt.title('Tsys: H pol')
    plt.subplot(n_ants,2,2)
    plt.title('Tsys: V pol')

    fig1.savefig(pp,format='pdf')
    plt.close(fig1)
    fig2.savefig(pp,format='pdf')
    plt.close(fig2)
    pp.close() # close the pdf file

# test main method for the library
if __name__ == "__main__":
#test the method with a know file
    filename = '/var/kat/archive/data/comm/2014/02/27/1393504489.h5'
    out = '.'
    band = 200e6
    print 'Performing test run with: ' + filename
    read_and_plot_data(filename,output_dir = out, freq_band=band)
