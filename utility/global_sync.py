#!/usr/bin/python
# Global sync with all the trimmings
#
# Initial script by Benjamin for RTS
# Updated for AR1 by Ruby
# Benjamin added PPS offsets
# Added timestamp of sync output for easier use
# Tiyani replaced dig_sync_epoch with dig_l_band_time_sync
# Reset capture destination
# Cleanup by Martin to remove redundant instructions now handled by CAM
# Tiyani: wait for dmc to update epoch before resetting capture destinations and
#         querying digitiser epoch
# Ruby stabilising script by exiting cleanly and allowing enough time for sensor updates
# Anton: limit cam object to components required, error on timeout, minor cleanup

from __future__ import with_statement

import time

from concurrent.futures import TimeoutError

from katcorelib import standard_script_options, verify_and_connect
from katcorelib import cambuild
import katconf


# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(usage="usage: %prog [options]",
                                 description="MeerKAT Global Sync Script ver 2\n" +
                                 "Performs a global sync,\n" +
                                 "Starts data stream from digitisers,\n" +
                                 "Resets capture destination to clear IP assignments")
parser.add_option('--configdelayfile', type="string",
                  default='katconfig/user/delay-models/mkat/pps_delays.csv',
                  help='Specify the katconfig path to the csv file containing receptor '
                       'delays in the format m0xx, <delay> (default="%default")')
parser.add_option('--mcpsetband', type="string", default='',
                  help='If specified, script will call cam.mcp.req.set_band() '
                       'with given parameter (default="%default")')
parser.add_option('--all', action="store_true", default=False,
                  help='Include all antennas in the global sync')
# assume basic options passed from instruction_set
parser.set_defaults(description='MeerKAT Global sync')
(opts, args) = parser.parse_args()
print("global_sync_MeerKAT script: start")

if opts.mcpsetband and opts.mcpsetband != 'l':
    raise RuntimeError('Unavailable band: mcpsetband has been specified as %s'
                       % opts.mcpsetband)

with verify_and_connect(opts) as kat:
    print("_______________________")
    print(kat.controlled_objects)
    print(kat.ants.clients)
    print(opts)
    print("_______________________")

    subarrays = kat.katpool.sensor.subarrays.get_value()
    subarrays_free = kat.katpool.sensor.subarrays_free.get_value()
    assert subarrays == subarrays_free, ("Please free all subarrays before "
                                         "running this script.")
    try:
        cam = None

        if not kat.dry_run:
            print('Building CAM object')
            ant_names = [ant.name for ant in kat.ants]
            components = ['mcp'] + ant_names
            cam = cambuild(password="camcam", full_control="all",
                           require=components, conn_clients=components)
            try:
                cam.until_synced(timeout=90)
            except TimeoutError:
                print("ERROR: Could not sync CAM container. ")
                for comp in cam.children.values():
                    if not comp.synced:
                        print('  CAM component not synced: ', comp.name)
                raise RuntimeError("CAM clients did not sync.")

            delay_list = {}
            try:
                delay_values = katconf.resource_string(opts.configdelayfile).split('\n')
                for line in delay_values:
                    x = ((line.strip('\n')).split(','))
                    if (len(x[0]) == 4 and x[0][0] == 'm'):
                        delay_list[x[0]] = int(x[1])
                for ant in sorted(delay_list):
                    print('Receptor: %s  delay: %s' % (ant, delay_list[ant]))
            except Exception as exc:
                raise RuntimeError('Failed to read pps delay file from config! '
                                   'File: {}.  Exception: {}.'
                                   .format(opts.configdelayfile, exc))

            if opts.all:
                ants_active = list(cam.ants)
            else:
                ants_active = [ant for ant in cam.ants if ant.name not in
                               kat.katpool.sensor.resources_in_maintenance.get_value()]
            ants_active.sort(key=lambda ant: ant.name)

            print('Set PPS delay compensation for digitisers')
            for ant in ants_active:
                # look at current delay and program in delay specified in CSV
                if ant.name in delay_list:
                    # set the delay compensations for a digitiser (assuming L band)
                    try:
                        response = ant.req.dig_digitiser_offset('l')
                    except Exception as msg:
                        print('Caught exception antenna %s' % ant.name)
                        print(msg)
                        raise

                    curr_delay_l = int(str(response).split(' ')[2])
                    if curr_delay_l == delay_list[ant.name]:
                        print(ant.name + ': no change to PPS delay offset')
                    else:
                        print(ant.name + " is on L band")
                        print(ant.name + ' L-band current delay : ' + str(curr_delay_l))
                        response = ant.req.dig_digitiser_offset('l', delay_list[ant.name])
                        print(ant.name + ' L-band PPS delay offset : ' + str(response))

            init_epoch = cam.mcp.sensor.dmc_synchronisation_epoch.get_value()
            print('Performing global sync on MeerKAT ...')
            serial_sync_timeout = 300  # seconds
            start_time = time.time()
            cam.mcp.req.dmc_global_synchronise(timeout=serial_sync_timeout)
            print("Duration of global sync: {}".format(time.time() - start_time))

            print('Previous sync time %d, waiting for new sync time' % init_epoch)
            cam_sleep = 2  # seconds to wait for CAM sensor retry
            wait_time = 0  # seconds
            while cam.mcp.sensor.dmc_synchronisation_epoch.get_value() == init_epoch:
                time.sleep(cam_sleep)
                wait_time += cam_sleep
                if wait_time >= 300:  # seconds
                    raise RuntimeError("dmc could not sync, investigation is required...")

            print('Setting digitiser L-band sync epoch to DMC epoch')
            dmc_epoch = cam.mcp.sensor.dmc_synchronisation_epoch.get_value()
            for ant in ants_active:
                try:
                    print("Verify digitiser epoch for antenna %s" % ant.name)
                    epoch_sensor = ant.sensor.dig_l_band_time_synchronisation_epoch
                    dig_sleep = 2  # seconds
                    wait_time = 0  # seconds
                    while epoch_sensor.get_value() != dmc_epoch:
                        time.sleep(dig_sleep)
                        wait_time += dig_sleep
                        if wait_time >= 60:  # seconds
                            print ("ant %s could not sync with DMC, investigation "
                                   "is required...!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                                   % ant.name)
                            break
                except Exception:
                    pass
                print("  %s sync epoch:  %d" % (ant.name, epoch_sensor.get_value()))
                print("  Resetting capture destination %s" % ant.name)
                ant.req.deactivate()
                capture_list = str(ant.req.dig_capture_list())
                for line in capture_list.splitlines():
                    print('    %s' % line)
            print('\n')
            print("Script complete")
    finally:
        if cam:
            print("Cleaning up cam object")
            cam.disconnect()

    print("\nGlobal Sync Date %s" % time.ctime(dmc_epoch))

# -fin-
