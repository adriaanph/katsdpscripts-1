#!/usr/bin/python
# Track target(s) for a specified time.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint

# temporary hack to ensure antenna does not timeout for the moment
def bad_ar1_alt_hack(target, duration, limit=88.):
    import numpy
    [az, el] = target.azel()
    delta_transit = duration*(15./3600.)
    if (numpy.rad2deg(float(el))+delta_transit+delta_transit) > limit: return True
    return False

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Track one or more sources for a specified time. At least one '
                                             'target must be specified. Note also some **required** options below.')
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=60.0,
                  help='Length of time to track each source, in seconds (default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='Maximum duration of the script in seconds, after which script will end '
                       'as soon as the current track finishes (no limit by default)')
parser.add_option('--repeat', action="store_true", default=False,
                  help='Repeatedly loop through the targets until maximum duration (which must be set for this)')

# Set default value for any option (both standard and experiment-specific options)
# parser.set_defaults(description='Target track',dump_rate=0.1)
parser.set_defaults(description='Target track')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name ('Cygnus A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    observation_sources = collect_targets(kat, args)
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
        user_logger.error("Dry Run: Unable to set Antenna mode to 'STOP'.")

    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
        # Start capture session, which creates HDF5 file
        with start_session(kat, **vars(opts)) as session:
            session.standard_setup(**vars(opts))
            session.capture_start()

            start_time = time.time()
            targets_observed = []
            # Keep going until the time is up
            keep_going = True
            while keep_going:
                keep_going = (opts.max_duration is not None) and opts.repeat
                targets_before_loop = len(targets_observed)
                # Iterate through source list, picking the next one that is up
                for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
# RvR -- Very bad hack to keep from tracking above 89deg until AR1 AP can handle out of range values better
#		    if bad_ar1_alt_hack(target, opts.track_duration):
#		        user_logger.info('Too high elevation, skipping target %s...' % target.name)
#                        user_logger.info("Target Az/El coordinates '%s'" % (str(target.azel())))
#			continue
# RvR -- Very bad hack to keep from tracking above 89deg until AR1 AP can handle out of range values better

                    session.label('track')
                    user_logger.info("Initiating %g-second track on target '%s'" % (opts.track_duration, target.name,))
# # RvR -- Debug output to try and track down timeout due to pointing
#                     user_logger.info("Target Az/El coordinates '%s'" % (str(target.azel())))
# # RvR -- Debug output to try and track down timeout due to pointing
                    # Split the total track on one target into segments lasting as long as the noise diode period
                    # This ensures the maximum number of noise diode firings
                    total_track_time = 0.
                    while total_track_time < opts.track_duration:
                        next_track = opts.track_duration - total_track_time
                        # Cut the track short if time ran out
                        if opts.max_duration is not None:
                            next_track = min(next_track, opts.max_duration - (time.time() - start_time))
                        if opts.nd_params['period'] > 0:
                            next_track = min(next_track, opts.nd_params['period'])
                        if next_track <= 0 or not session.track(target, duration=next_track, announce=False):
                            break
                        total_track_time += next_track
                    if opts.max_duration is not None and (time.time() - start_time >= opts.max_duration):
                        user_logger.warning("Maximum duration of %g seconds has elapsed - stopping script" %
                                            (opts.max_duration,))
                        keep_going = False
                        break
                    targets_observed.append(target.name)
                if keep_going and len(targets_observed) == targets_before_loop:
                    user_logger.warning("No targets are currently visible - stopping script instead of hanging around")
                    keep_going = False
            user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))

# RvR -- Temporary measure to put antennas in stop mode until we can go back to safe stow positions
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
        user_logger.error("Dry Run: Unable to set Antenna mode to 'STOP'.")
# RvR -- Temporary measure to put antennas in stop mode until we can go back to safe stow positions
