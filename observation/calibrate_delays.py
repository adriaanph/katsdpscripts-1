#!/usr/bin/env python
#
# Track delay calibrator target for a specified time.
# Obtain delay solutions and apply them to the delay tracker in the CBF proxy.
#
# Ludwig Schwardt
# 5 April 2017
#

import json
import time

from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger)


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


class NoDelaysAvailableError(Exception):
    """No delay solutions are available from the cal pipeline."""


# Default F-engine gain as a function of number of channels
DEFAULT_GAIN = {4096: 200, 32768: 4000}


def set_fengine_gain(session, gain):
    """Set F-engine gain to *gain* if positive, or automatic default if 0."""
    if session.kat.dry_run:
        gain = -1
    # Obtain default gain based on channel count if none specified
    if gain == 0:
        num_channels = session.cbf.fengine.sensor.n_chans.get_value()
        gain = DEFAULT_GAIN.get(num_channels, -1)
    if gain > 0:
        user_logger.info("Setting F-engine gains to %d" % (gain,))
        for inp in session.cbf.fengine.inputs:
            session.cbf.fengine.req.gain(inp, gain)


def get_cal_inputs(telstate):
    """Input labels associated with calibration products."""
    if 'cal_antlist' not in telstate or 'cal_pol_ordering' not in telstate:
        return []
    ants = telstate['cal_antlist']
    polprods = telstate['cal_pol_ordering']
    pols = [prod[0] for prod in polprods if prod[0] == prod[1]]
    return [ant + pol for pol in pols for ant in ants]


def get_delaycal_solutions(session):
    """Retrieve delay calibration solutions from telescope state."""
    inputs = get_cal_inputs(session.telstate)
    if not inputs or 'cal_product_K' not in session.telstate:
        return {}
    solutions, solution_ts = session.telstate.get_range('cal_product_K')[0]
    if solution_ts < session.start_time:
        return {}
    # XXX katsdpcal currently has solutions the wrong way around
    solutions = -solutions
    return dict(zip(inputs, solutions.real.flat))


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Track the source with the highest elevation and calibrate ' \
              'delays based on it. At least one target must be specified.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=30.0,
                  help='Length of time to track the source, in seconds (default=%default)')
parser.add_option('--fengine-gain', type='int', default=0,
                  help='Correlator F-engine gain, automatically set if 0 (the '
                       'default) and left alone if negative')
parser.add_option('--reset-delays', action='store_true', default=False,
                  help='Zero the delay adjustments afterwards')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(observer='comm_test', nd_params='off', project_id='COMMTEST',
                    description='Delay calibration observation')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name "
                     "('Cygnus A'), description ('azel, 20, 30') or catalogue "
                     "file name ('sources.csv')")

# Check options and build KAT configuration, connecting to proxies and clients
with verify_and_connect(opts) as kat:
    observation_sources = collect_targets(kat, args)
    # Quit early if there are no sources to observe
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        raise NoTargetsUpError("No targets are currently visible - "
                               "please re-run the script later")
    # Pick source with the highest elevation as our target
    target = observation_sources.sort('el').targets[-1]
    target.add_tags('bfcal single_accumulation')
    # Start capture session
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        set_fengine_gain(session, opts.fengine_gain)
        user_logger.info("Zeroing all delay adjustments for starters")
        session.cbf.req.adjust_all_delays()
        session.capture_start()
        user_logger.info("Initiating %g-second track on target %r",
                         opts.track_duration, target.description)
        session.label('un_corrected')
        session.track(target, duration=opts.track_duration, announce=False)
        # Attempt to jiggle cal pipeline to drop its delay solutions
        session.ants.req.target('')

        user_logger.info("Waiting for delays to materialise in cal pipeline")
        time.sleep(30)
        sample_rate = 0.0
        delays = {}
        if not kat.dry_run:
            sample_rate = session.telstate.get('cbf_adc_sample_rate', 0.0)
            delays = get_delaycal_solutions(session)
            # JSON does not like NumPy types
            delays = {inp: float(d) for inp, d in delays.items()}
            if not delays:
                msg = "No delay solutions found in telstate '%s'" % \
                      (session.telstate,)
                raise NoDelaysAvailableError(msg)
        user_logger.info("Delay solutions (sample rate = %g Hz):", sample_rate)
        for inp in sorted(delays):
            user_logger.info(" - %s: %10.3f ns, %9.2f samples",
                             inp, delays[inp] * 1e9, delays[inp] * sample_rate)
        user_logger.info("Adjusting delays on CBF proxy")
        session.cbf.req.adjust_all_delays(json.dumps(delays))

        user_logger.info("Revisiting target %r for %g seconds to see if "
                         "delays are fixed", target.name, opts.track_duration)
        session.label('corrected')
        session.track(target, duration=opts.track_duration, announce=False)
        if opts.reset_delays:
            user_logger.info("Zeroing all delay adjustments on CBF proxy")
            session.cbf.req.adjust_all_delays()
