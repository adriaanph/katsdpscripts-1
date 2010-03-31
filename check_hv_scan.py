#!/usr/bin/python
# Perform tipping curve scan for a specified azimuth position.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import optparse
import sys
import uuid
import time
import numpy as np
import katuilib

# Parse command-line options that allow the defaults to be overridden
parser = optparse.OptionParser(usage="%prog [options]",
                               description="Perform tipping curve scan for a specified azimuth position. \
                                            Some options are **required**.")
# Generic options
parser.add_option('-i', '--ini_file', dest='ini_file', type="string", metavar='INI', help='Telescope configuration ' +
                  'file to use in conf directory (default reuses existing connection, or falls back to cfg-local.ini)')
parser.add_option('-s', '--selected_config', dest='selected_config', type="string", metavar='SELECTED',
                  help='Selected configuration to use (default reuses existing connection, or falls back to local_ff)')
parser.add_option('-u', '--experiment_id', dest='experiment_id', type="string",
                  help='Experiment ID used to link various parts of experiment together (UUID generated by default)')
parser.add_option('-o', '--observer', dest='observer', type="string",
                  help='Name of person doing the observation (**required**)')
parser.add_option('-d', '--description', dest='description', type="string", default="Tipping curve",
                  help='Description of observation (default="%default")')
parser.add_option('-a', '--ants', dest='ants', type="string", metavar='ANTS',
                  help="Comma-separated list of antennas to include in scan (e.g. 'ant1,ant2')," +
                       " or 'all' for all antennas (**required** - safety reasons)")
parser.add_option('-f', '--centre_freq', dest='centre_freq', type="float", default=1822.0,
                  help='Centre frequency, in MHz (default="%default")')
parser.add_option('-w', '--discard_slews', dest='record_slews', action="store_false", default=True,
                  help='Do not record all the time, i.e. pause while antennas are slewing to the next target')
# Experiment-specific options
parser.add_option('-z', '--az', dest='az', type="float", default=168.0,
                  help='Azimuth angle along which to do tipping curve, in degrees (default="%default")')

(opts, args) = parser.parse_args()

# Various non-optional options...
if opts.ants is None:
    print 'Please specify the antennas to use via -a option (yes, this is a non-optional option...)'
    sys.exit(1)
if opts.observer is None:
    print 'Please specify the observer name via -o option (yes, this is a non-optional option...)'
    sys.exit(1)
if opts.experiment_id is None:
    # Generate unique string via RFC 4122 version 1
    opts.experiment_id = str(uuid.uuid1())

# Try to build the given KAT configuration (which might be None, in which case try to reuse latest active connection)
# This connects to all the proxies and devices and queries their commands and sensors
try:
    kat = katuilib.tbuild(opts.ini_file, opts.selected_config)
# Fall back to *local* configuration to prevent inadvertent use of the real hardware
except ValueError:
    kat = katuilib.tbuild('cfg-local.ini', 'local_ff')
print "\nUsing KAT connection with configuration: %s\n" % (kat.get_config(),)

# start up the signal display handler so we have access to the data...
#kat.dh.start_sdisp()

# get some antennas to work with
ants = katuilib.observe.ant_array(kat, opts.ants)
 # set the centre freq
kat.rfe7.req.rfe7_lo1_frequency(4200.0 + opts.centre_freq, 'MHz')

sigs = kat.dh.sd.cpref._real_to_dbe.keys()

kat.dbe.req.capture_stop()

ants.req.target('azel, %f, %f' % (opts.az, 2))
 # on losberg
ants.req.mode("POINT")
ants.wait("lock",1)
 # wait for them to get there

kat.dbe.req.capture_setup(1000,1.8)
kat.dbe.req.capture_start()

time.sleep(30)
 # get some data
don = {}
for s in sigs:
    don[s] = np.array(kat.dh.sd.select_data(product=(int(s[:-1]),int(s[:-1]),s[-1:]+s[-1:]), dtype='mag', start_channel=100, stop_channel=400, end_time=-15))

ants.req.target('azel, %f, %f' % (opts.az, 20))
 # cold(ish) sky
ants.wait("lock",1)
time.sleep(30)

doff = {}
sigs.sort()
for s in sigs:
    doff[s] = np.array(kat.dh.sd.select_data(product=(int(s[:-1]),int(s[:-1]),s[-1:]+s[-1:]), dtype='mag', start_channel=100, stop_channel=400, end_time=-15))
    print "For real input",s,"the mean on source / off source ratio is:",don[s].mean() / doff[s].mean()

for s in sigs:
    print doff[s].mean(axis=1)
    print don[s].mean(axis=1)

kat.dbe.req.capture_stop()
ants.req.mode("STOP")
