#!/usr/bin/python
# Track target(s) for a specified time.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement
import numpy as np
import time
from katcorelib import collect_targets, standard_script_options, verify_and_connect,  start_session, user_logger
import katpoint

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Track one or more sources for a specified time. At least one '
                                             'target must be specified. Note also some **required** options below.')
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=20.0,
                  help='Length of time to track each source, in seconds (default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='Maximum duration of the script in seconds, after which script will end '
                       'as soon as the current track finishes (no limit by default)')
parser.add_option( '--max-extent', type='float', default=1.0,
                  help='Maximum extent in degrees, the script will scan ')
parser.add_option( '--number-of-steps', type='int', default=10,
                  help='Number of pointings to do while scaning , the script will scan ')
parser.add_option('--no-delays', action="store_true", default=False,
                  help='Do not use delay tracking, and zero delays')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Inferometric Pointing offset track')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name ('Cygnus A'), "
                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    args_target_list =[]
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(10)
    else:
        user_logger.error("Unable to set Antenna mode to 'STOP'.")

    observation_sources = katpoint.Catalogue(antenna=kat.sources.antenna)
    
    try:
        observation_sources.add_tle(file(args[0]))
    except (IOError, ValueError):#IOError or ValueError : # If the file failed to load assume it is a target string
        args_target_obj = collect_targets(kat,args)
        observation_sources.add(args_target_obj)
            
            #user_logger.info("Found %d targets from Command line and %d targets from %d Catalogue(s) " %
            #                         (len(args_target_obj),num_catalogue_targets,len(args)-len(args_target_list),))
    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
        # Start capture session, which creates HDF5 file
        with start_session(kat, **vars(opts)) as session:
            if not opts.no_delays and not kat.dry_run :
                if session.dbe.req.auto_delay('on'):
                    user_logger.info("Turning on delay tracking.")
                else:
                    user_logger.error('Unable to turn on delay tracking.')
            elif opts.no_delays and not kat.dry_run:
                if session.dbe.req.auto_delay('off'):
                    user_logger.info("Turning off delay tracking.")
                else:
                    user_logger.error('Unable to turn off delay tracking.')
                #if session.dbe.req.zero_delay():
                #    user_logger.info("Zeroed the delay values.")
                #else:
                #    user_logger.error('Unable to zero delay values.')

            session.standard_setup(**vars(opts))
            session.capture_start()

            start_time = time.time()
            targets_observed = []
            # Keep going until the time is up
            keep_going = True
            while keep_going:
                keep_going = (opts.max_duration is not None)
                targets_before_loop = len(targets_observed)
                # Iterate through source list, picking the next one that is up
                for target in observation_sources.iterfilter(el_limit_deg=opts.horizon):
                    session.set_target(target) # Set the target
                    session.label('pointing_cluster')
                    session.track(target, duration=opts.track_duration, announce=False) # Set the target & mode = point
                    for direction  in   {'x','y'} :
                        for offset in np.linspace(-opts.max_extent,opts.max_extent,opts.number_of_steps//2):
                            if direction == 'x' :
                                offset_target = [offset,0.0]
                            else:
                                offset_target = [0.0,offset]
                            user_logger.info("Initiating %g-second track on target '%s'" % (opts.track_duration, target.name,))
                            user_logger.info("Offset of %f,%f degrees " %(offset_target[0],offset_target[1]))
                            session.set_target(target)
                            if not kat.dry_run :
                                session.ants.req.offset_fixed(offset_target[0],offset_target[1],opts.projection)
                            nd_params = session.nd_params
                            session.fire_noise_diode(announce=True, **nd_params)
                            if not kat.dry_run :
                                time.sleep(opts.track_duration) # Snooze
                            else:
                                session.track(target, duration=opts.track_duration, announce=False)
                                # Just work out dry run time
                                
                    targets_observed.append(target.name)
                    if opts.max_duration is not None and (time.time() - start_time >= opts.max_duration):
                        user_logger.warning("Maximum duration of %g seconds has elapsed - stopping script" %
                                            (opts.max_duration,))
                        keep_going = False
                        break
            
                if keep_going and len(targets_observed) == targets_before_loop:
                    user_logger.warning("No targets are currently visible - stopping script instead of hanging around")
                    keep_going = False
            user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))
