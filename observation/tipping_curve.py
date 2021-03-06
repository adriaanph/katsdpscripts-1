#!/usr/bin/env python
# Perform tipping curve scan  and find a specified azimith if one is not give.

import time

from katcorelib import (standard_script_options, verify_and_connect,
                        start_session, user_logger)
import numpy as np


# Set up standard script options
parser = standard_script_options(usage="%prog [options]",
                                 description="Perform tipping curve scan for a specified azimuth position. \
                                              Or Select a Satilite clear Azimith,\
                                              Some options are **required**.")
# Add experiment-specific options
parser.add_option('-z', '--az', type="float", default=None,
                  help='Azimuth angle along which to do tipping curve, in degrees (default="%default")')
parser.add_option('--spacing', type="float", default=1.0,
                  help='The Spacing along the elevation axis of the tipping curve '
                       'that measurements are taken, in degrees (default="%default")')
parser.add_option('--tip-both-directions', action="store_true", default=False,
                  help='Do tipping curve from low to high elevation and then from high to low elevation')
parser.add_option('--max-elevation', type='float', default=90.0,
                  help="Maximum elevation angle, in degrees (default=%default)")
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Tipping Curve')
# Parse the command line
opts, args = parser.parse_args()

on_time = 15.0
# Iterate through elevation angles
spacings = list(np.arange(opts.horizon, opts.max_elevation + 0.1, opts.spacing))
if opts.tip_both_directions:
    spacings += list(np.arange(opts.max_elevation, opts.horizon - 0.1, -opts.spacing))

with verify_and_connect(opts) as kat:
    # Ensure that azimuth is in valid physical range of -185 to 275 degrees
    if opts.az is None:
        user_logger.info("No Azimuth selected, selecting clear Azimith")
        timestamp = time.time() + np.arange(len(spacings)) * (on_time + 20.0 + 1.0)
        if not kat.dry_run:
            # timestamp = [katpoint.Timestamp(time.time()) for i in range(int((np.arange(
            #      opts.horizon,opts.max_elevation+0.1,opts.spacing).shape[0]*(on_time+20.0+1.0))))]
            # Load the standard KAT sources ... similar to the SkyPlot of the katgui
            sources_to_avoid = kat.sources
            source_az = []
            for source in sources_to_avoid.targets:
                az, el = np.degrees(source.azel(timestamp=timestamp))   # was rad2deg
                az[az > 180] = az[az > 180] - 360
                source_az += list(set(az[el > opts.horizon]))
            source_az.sort()
            gap = np.diff(source_az).argmax() + 1
            opts.az = (source_az[gap] + source_az[gap + 1]) / 2.0
            user_logger.info("Selecting Satellite clear Azimuth=%f", opts.az)
        else:
            opts.az = 0
            user_logger.info("Selecting dummy Satellite clear Azimuth=%f", opts.az)
    else:
        if (opts.az < -185.) or (opts.az > 275.):
            opts.az = (opts.az + 180.) % 360. - 180.
        user_logger.info("Tipping Curve at Azimuth=%f", opts.az)
    user_logger.info("Tipping Curve at Azimuth=%f", opts.az)

    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()
#         # Iterate through elevation angles
#         spacings = list(np.arange(opts.horizon,opts.max_elevation+0.1,opts.spacing))
#         if opts.tip_both_directions :
#             spacings += list(np.arange(opts.max_elevation,opts.horizon-0.1,-opts.spacing))
        for el in spacings:
            session.label('track')
            session.track('azel, %f, %f' % (opts.az, el), duration=on_time)
            session.fire_noise_diode('coupler', on=10, off=10)
