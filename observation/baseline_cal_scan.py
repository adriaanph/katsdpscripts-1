#!/usr/bin/env python
# Track various point sources as specified in a catalogue
# for the purpose of baseline calibration.

import time

from katcorelib import (standard_script_options, verify_and_connect,
                        collect_targets, start_session, user_logger)
import katpoint


# Set up standard script options
description = 'Track various point sources specified by name, string or ' \
              'catalogue, or use the default catalogue if none are ' \
              'specified. This is useful for baseline (antenna location) ' \
              'calibration. Remember to specify the observer and antenna ' \
              'options, as these are **required**.'
parser = standard_script_options(usage="%prog [options] [<'target/catalogue'> ...]",
                                 description=description)
# Add experiment-specific options
parser.add_option('-m', '--min-time', type='float', default=-1.0,
                  help="Minimum duration to run experiment, in seconds "
                       "(default=one loop through sources)")
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Baseline calibration', nd_params='off',
                    no_delays=True)
# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:
    # Create baseline calibrator catalogue
    if len(args) > 0:
        # Load catalogue files or targets if given
        baseline_sources = collect_targets(kat, args)
    else:
        # Prune the standard catalogue to only contain sources that
        # are good for baseline calibration
        great_sources = ['3C123', 'Taurus A', 'Orion A', 'Hydra A', '3C273',
                         'Virgo A', 'Centaurus A', 'Pictor A']
        good_sources = ['3C48', '3C84', 'J0408-6545', 'J0522-3627', '3C161',
                        'J1819-6345', 'J1939-6342', '3C433', 'J2253+1608']
        baseline_sources = katpoint.Catalogue([kat.sources[src] for src in great_sources + good_sources],
                                              antenna=kat.sources.antenna)
        user_logger.info("No targets specified, loaded default catalogue with %d targets",
                         len(baseline_sources))

    with start_session(kat, **vars(opts)) as session:
        # Force delay tracking to be off
        opts.no_delays = True
        session.standard_setup(**vars(opts))
        session.capture_start()
        start_time = time.time()
        targets_observed = []
        # Keep going until the time is up
        keep_going = True
        while keep_going:
            # Iterate through baseline sources that are up
            for target in baseline_sources.iterfilter(el_limit_deg=opts.horizon):
                session.label('track')
                session.track(target, duration=120.0)
                targets_observed.append(target.name)
                # The default is to do only one iteration through source list
                if opts.min_time <= 0.0:
                    keep_going = False
                # If the time is up, stop immediately
                elif time.time() - start_time >= opts.min_time:
                    keep_going = False
                    break
        user_logger.info("Targets observed : %d (%d unique)",
                         len(targets_observed), len(set(targets_observed)))
