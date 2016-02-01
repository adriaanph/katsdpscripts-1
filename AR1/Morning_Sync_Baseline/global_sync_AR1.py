#!/usr/bin/python
# Global sync with all the trimmings
#
# Initial script by Benjamin for RTS
# Updated for AR1 by Ruby -- will use print to display both in GUI and on Ipython

from __future__ import with_statement
import time, string
from katcorelib import standard_script_options, verify_and_connect, user_logger, start_session
from katcorelib import cambuild, katconf

# Parse command-line options that allow the defaults to be overridden
parser = standard_script_options(usage="usage: %prog [options]",
                            description="AR1 Global Sync Script ver 2\n"+
                            "Performs a global sync,\n"+
                            "Starts data stream from digitisers,\n"+
                            "Halts AR1 array and programs the correlator")
# assume options passed from instruction_set
parser.set_defaults(description = 'AR1 Global sync')
(opts, args) = parser.parse_args()
print("global_sync_AR1 script: start")

def log_info(response):
    response = str(response)
    if 'fail' in response:
        user_logger.warn(response)
    else:
        user_logger.info(response)

with verify_and_connect(opts) as kat:
    print "_______________________"
    print kat.controlled_objects
    print kat.ants.clients
    print opts
    print "_______________________"
    try:
        cam = None
        cont = False
        count = 1

        if not kat.dry_run:
	    print('Building CAM object')
            cam = cambuild(password="camcam", full_control="all")
	    time.sleep(5)

	    print('Performing global sync on AR1 ...')
	    cam.mcp.req.dmc_global_synchronise(timeout=30)
	    time.sleep(5)

	    print('Reiniting all digitisers ...')
	    ant_active = [ant for ant in cam.ants if ant.name not in cam.katpool.sensor.resources_in_maintenance.get_value()]
	    antlist=''
	    for ant in ant_active:
	        if antlist: antlist=','.join((antlist,ant.name))
	        else: antlist=ant.name
	        response = ant.req.dig_capture_start('hv')
	        print(ant.name + ': ' + str(response))
	        time.sleep(1)

# RvR -- For the moment assume always subarray_1 -- need to follow up with cam about knowing which is active
	    print('Halting ar1 array...')
	    cam.subarray_1.req.free_subarray(timeout=30)
	    print('Waiting 5 seconds for things to settle')
	    time.sleep(10)

	    corrprod = opts.product
	    if corrprod not in ('c856M4k', 'c856M32k'):
	        corrprod = 'c856M4k'
	        print('No correlation product specified. Using %s' % corrprod)

            while not cont:
		print('Building new subarray, this may take a little time....')
 		cam.subarray_1.req.set_band('l')
                cam.subarray_1.req.set_product(corrprod)
 		cam.subarray_1.req.assign_resources('data_1,'+antlist)
		response=cam.subarray_1.req.activate_subarray(timeout=300)
# RvR -- For the moment assume always subarray_1 -- need to follow up with cam about knowing which is active

		if 'ok' in str(response):
                    cont = True
                    print('Programming correlator successful!')
                    print('Subarray 1 active!')
                else:
                    count = count + 1
                    print('Failure to program correlator!!!  Trying again.....')
	        time.sleep(2)

                if count > 5:
		    print('Cannot auto-activate subarray, giving up.....')
		    break

            time.sleep(5)

            print("Script complete")
    finally:
        if cam:
	    print("Cleaning up cam object")
            cam.disconnect()

# -fin-
