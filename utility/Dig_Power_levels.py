#run script : python script.py
#! bin/usr/python
import requests
import time
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


url="http://portal.mkat.karoo.kat.ac.za/katstore/samples" # site for the most resent values
#url='http://kat-flap-cpt.mkat.control.kat.ac.za/katstore/samples' # flap in CT for older values
def getAnts(sensor, url,timest=time.time()):
    """Get the number of antennas in MeerKAT 900 seconds ago by querying the katstore url using each antenna CAM sensor"""
    sensor_info={
                    'sensor': sensor,
                    'start': timest - 900,
                    'end':  timest,
                    'limit':1,
                 }
    res = requests.get(url, params=sensor_info)
    katsi = re.compile('m\d{3}')
    ants = re.findall(katsi, res.content)
    return ants

all_ants = getAnts('katpool_ants', url)
maint_ants=list(sorted(getAnts('katpool_resources_in_maintenance',url)))
receptors =list(sorted(all_ants))
active_receptors= set(receptors)-set(maint_ants)

print('MeerKAT receptors: {}\n\n{}'.format(len(receptors), receptors))
print ('Receptors in Maintanance:{}\n\n{}'.format(len(maint_ants), maint_ants))
print ('Active receptors {}\n\n{}'.format(len(active_receptors), active_receptors))

def getSensorValues(ants,timest=time.time()):
    """ Get digitiser sensor values for each antenna and store values into a structured array"""

    Store_array=np.zeros((len(ants),1), dtype=([('Antennas','S5'),
                                        ('dig_selected_band','S5'),
                                        ('rsc_rxl_serial_number', 'S5'),
                                        ('dig_l_band_rfcu_hpol_rf_power_in', 'S5'),
                                        ('dig_l_band_rfcu_vpol_rf_power_in','S5'),
                                        ('dig_l_band_adc_hpol_rf_power_in','S5'),
                                        ('dig_l_band_adc_vpol_rf_power_in','S5'),
                                        ('dig_l_band_rfcu_hpol_attenuation',int),
                                        ('dig_l_band_rfcu_vpol_attenuation',int)]))
    keydict={'dig_selected_band',
             'rsc_rxl_serial_number',
             'dig_l_band_rfcu_hpol_rf_power_in',
             'dig_l_band_rfcu_vpol_rf_power_in',
             'dig_l_band_adc_hpol_rf_power_in',
             'dig_l_band_adc_vpol_rf_power_in',
             'dig_l_band_rfcu_hpol_attenuation',
             'dig_l_band_rfcu_vpol_attenuation'}

    for i in range(len(ants)):
        value=dict([(key, []) for key in keydict])

        ant=ants[i]
        Store_array[i]['Antennas']=ant

        for key in keydict:
            res = requests.get(url, params={'sensor': ant+'_'+key,
                           'start': timest - 900,
                           'end':  timest,
                           'limit':1})

            value[key].append(res.content.split(',')[3][1:-1])
            Store_array[i][key]=value[key]

        print "Processing sensor information for antenna %s" % ant

    return Store_array

def rfcu_calc(rfcuinPol,rfcuoutPol,atten):
    """rcuinPol is a list of power in to digitser to the ADC per antenna and polarization (rfcu.pol.rf.power.in)
      rfcuoutPol is a list of power out of the ADC per antenna and polarization  (adc.pol.rf.power.in)
      atten is a list of attenuation levels per antenna and polarization"""
    #input pwr + 14dB (or 16dB) - atten = (-40+14-6)
    rfcuCalc=np.array(rfcuinPol)+16-atten
    Amp_gain=abs(np.array(rfcuinPol))-(abs(np.array(rfcuoutPol))-atten)
    return  rfcuCalc,Amp_gain

#target_rfcuin=-40
#target_rfcuout_min=-30
#target_rfcout_max=-24
def plot_rfcuPol(AmpGainPol,rfcuinPol,rfcuoutPol,rfcuCalcPol):
    """AmpGainPol is an array of Amplification gain estimated by subtracting the power levels reported by the adc.pol.rf.power.in
    and the rfcu.pol.rf.power.in
    rfcuCalcPol is an array of Calculated adc-in using the equation input pwr + 14dB (or 16dB) - atten """
    fig=plt.figure(figsize=(20,12))
    ax1=plt.subplot(211)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.plot(AmpGainPol,'mo-',lw=2,label='Amp gain')
    ax1.set_ylabel('Amplification Gain(dB)',fontsize=20, fontweight='bold')
    ax1.axhline(y=13, color='y',lw=1.5, linestyle='--',label='Min amp gain')
    ax1.axhline(y=16, color='orange',lw=1.5, linestyle='--',label='Max amp gain')
    plt.yticks(size=15)
    ax1.grid(linestyle='dotted')
    ax1.legend(fontsize=15,loc='upper center', bbox_to_anchor=(1.0, 1.),
          fancybox=True, shadow=True)
    ax2=plt.subplot(212,sharex=ax1)
    ax2.set_ylabel('Power(dB)',fontsize=20, fontweight='bold')


    ax2.plot(rfcuinPol,'bo-',lw=2,label='rfcu-in')
    ax2.plot(rfcuoutPol,'ro-',lw=2,label='system adc-in')
    ax2.plot(rfcuCalcPol,'co-',lw=2,label='rfcu-in+16dB-atten')
    ax2.axhline(y=-40, color='b',lw=1.5, linestyle='--',label='Design rfcu-in')
    ax2.axhline(y=-30, color='r',lw=1.5, linestyle='--',label='Design Min adc-in')
    ax2.axhline(y=-24, color='r',lw=1.5, linestyle='-',label='Max adc-in')
    ax2.legend(fontsize=15,loc='upper center', bbox_to_anchor=(1.0, 1.8),
          fancybox=True, shadow=True)
    ax2.grid(linestyle='dotted')
    plt.xticks(range(len(Antennas)), Antennas, rotation=60,size=15)
    plt.yticks(size=15)
    ax2.set_xlabel('Antenna',fontsize=20, fontweight='bold')
    plt.subplots_adjust(hspace=.1)
    return fig

"""Calling main funtion to store sensor values"""
data=getSensorValues(receptors)



"""Extracting H_pol rfcu-in and adc-in from data"""
rfcuin_H=[float(np.asscalar(i)) for i in data['dig_l_band_rfcu_hpol_rf_power_in']]
rfcuout_H=[float(np.asscalar(i)) for i in data['dig_l_band_adc_hpol_rf_power_in']]

"""Extracting  V_pol rfcu-in and adc-in from data"""
rfcuin_V=[float(np.asscalar(i)) for i in data['dig_l_band_rfcu_vpol_rf_power_in']]
rfcuout_V=[float(np.asscalar(i)) for i in data['dig_l_band_adc_vpol_rf_power_in']]

"""Extracting all the antennas And attenuation"""
Antennas=[np.asscalar(i) for i in data['Antennas']]
att_H=np.ravel(data['dig_l_band_rfcu_hpol_attenuation'])
att_V=np.ravel(data['dig_l_band_rfcu_vpol_attenuation'])

"""Calculating rfcu out"""
rfcuCalc_H=rfcu_calc(rfcuin_H,rfcuout_H,att_H)[0]
rfcuCalc_V=rfcu_calc(rfcuin_V,rfcuout_V,att_V)[0]
"""Calculate the amplification gain """
AmpGain_H=rfcu_calc(rfcuin_H,rfcuout_H,att_H)[1]
AmpGain_V=rfcu_calc(rfcuin_V,rfcuout_V,att_V)[1]
pp =PdfPages('Digitiser_power_levels'+'.pdf')

"""Calling plot functions H & V """
fig=plot_rfcuPol(AmpGain_H,rfcuin_H,rfcuout_H,rfcuCalc_H)
plt.title('Digitiser power level-Hpol',fontsize=20, fontweight='bold')
fig.savefig(pp,format='pdf')
plt.close(fig)


fig=plot_rfcuPol(AmpGain_V,rfcuin_V,rfcuout_V,rfcuCalc_V)
plt.title('Digitiser power level-Vpol',fontsize=20, fontweight='bold')
fig.savefig(pp,format='pdf')
plt.close(fig)
pp.close()
plt.close('all')
