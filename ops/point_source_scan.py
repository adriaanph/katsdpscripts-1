# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import katuilib
from katuilib import CaptureSession
import katpoint

import optparse
import sys
import uuid

# Parse command-line options that allow the defaults to be overridden
# Default KAT configuration is *local*, to prevent inadvertent use of the real hardware
parser = optparse.OptionParser(usage="%prog [options] [<catalogue file>]")
# Generic options
parser.add_option('-i', '--ini_file', dest='ini_file', type="string", default="cfg-local.ini", metavar='INI',
                  help='Telescope configuration file to use in conf directory (default="%default")')
parser.add_option('-s', '--selected_config', dest='selected_config', type="string", default="local_ff", metavar='SELECTED',
                  help='Selected configuration to use (default="%default")')
parser.add_option('-u', '--experiment_id', dest='experiment_id', type="string",
                  help='Experiment ID used to link various parts of experiment together (UUID generated by default)')
parser.add_option('-o', '--observer', dest='observer', type="string", help='Name of person doing the observation')
parser.add_option('-d', '--description', dest='description', type="string", default="Point source scan",
                  help='Description of observation (default="%default")')
parser.add_option('-a', '--ants', dest='ants', type="string", metavar='ANTS',
                  help="Comma-separated list of antennas to include in scan (e.g. 'ant1,ant2')," +
                       " or 'all' for all antennas - this MUST be specified (safety reasons)")
parser.add_option('-f', '--centre_freq', dest='centre_freq', type="float", default=1822.0,
                  help='Centre frequency, in MHz (default="%default")')
parser.add_option('-p', '--print_only', dest='print_only', action="store_true", default=False,
                  help="Do not actually observe, but display which sources will be scanned (default=%default)")
parser.add_option('-m', '--max_time', dest='max_time', type="float", default=-1.0,
                  help="Time limit on experiment, in seconds (default=no limit)")
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

# Build KAT configuration, as specified in user-facing config file
# This connects to all the proxies and devices and queries their commands and sensors
kat = katuilib.tbuild(opts.ini_file, opts.selected_config)

# Load pointing calibrator catalogues
if len(args) > 0:
    pointing_sources = katpoint.Catalogue(add_specials=False, antenna=kat.sources.antenna)
    for catfile in args:
        pointing_sources.add(file(catfile))
else:
    # Default catalogue contains the radec sources in the standard kat database
    pointing_sources = kat.sources.filter(tags='radec')

start_time = katpoint.Timestamp()

if opts.print_only:
    current_time = katpoint.Timestamp(start_time)
    # Find out where first antenna is currently pointing (assume all antennas point there)
    ants = katuilib.observe.ant_array(kat, opts.ants)
    az = ants.devs[0].sensor.pos_actual_scan_azim.get_value()
    el = ants.devs[0].sensor.pos_actual_scan_elev.get_value()
    prev_target = katpoint.construct_azel_target(az, el)
    # Only list targets that will be visited
    for compscan, target in enumerate(pointing_sources.iterfilter(el_limit_deg=5, timestamp=current_time)):
        print "At about %s, antennas will start slewing to '%s'" % (current_time.local(), target.name)
        # Assume 1 deg/s slew rate on average -> add time to slew from previous target to new one
        current_time += 1.0 * katpoint.rad2deg(target.separation(prev_target))
        print "At about %s, point source scan (compound scan %d) will start on '%s'" % \
              (current_time.local(), compscan, target.name)
        # Standard raster scan is 3 scans of 20 seconds each, with 2 slews of about 2 seconds in between scans,
        # followed by 10 seconds of noise diode on/off. Also allow one second of overhead per scan.
        current_time += 3 * 20.0 + 2 * 2.0 + 10.0 + 8 * 1.0
        prev_target = target
        if (opts.max_time > 0) and (current_time - start_time >= opts.max_time):
            break
    print "Experiment finished at about", current_time.local()

else:
    # The real experiment: Create a data capturing session with the selected sub-array of antennas
    with CaptureSession(kat, opts.experiment_id, opts.observer, opts.description, opts.ants, opts.centre_freq) as session:
        # While experiment time is not up
        while (opts.max_time <= 0) or (katpoint.Timestamp() - start_time < opts.max_time):
            # Iterate through source list, picking the next one that is up
            for target in pointing_sources.iterfilter(el_limit_deg=5):
                # Do standard raster scan on target
                session.raster_scan(target)
                # Fire noise diode, to allow gain calibration
                session.fire_noise_diode('coupler')

# WORKAROUND BEWARE
# Don't disconnect for IPython, but disconnect when run via standard Python
# Without this disconnect, the script currently hangs here when run from the command line
try:
    import IPython
    if IPython.ipapi.get() is None:
        kat.disconnect()
except ImportError:
    kat.disconnect()
