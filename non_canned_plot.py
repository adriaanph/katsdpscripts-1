#!/usr/bin/python
# Drive antenna to two targets and then make a pointing error plot 

import ffuilib as ffui
import numpy as np
import pylab

#ff = ffui.cbuild("ffuilib.ant_only.rc")
ff = ffui.tbuild("cfg-telescope.ini","local_ant_only")
 # make fringe fingder connections

ff.ant2.req_target_azel(20.31,30.45)
 # send an az/el target to antenna 2

ff.ant2.req_mode("POINT")
 # switch to mode point

ff.ant2.wait("lock","1",120)
 # wait for lock to be achieved (timeout=120 seconds)

ff.ant2.req_target_azel(40.2,60.32)
 # send a new az/el target

ff.ant2.wait("lock","1",120)
 # wait for lock again

 # produce custom pointing error plot
 # each sensor has local history

req_az = ff.ant2.sensor_pos_request_scan_azim.get_cached_history()
req_el = ff.ant2.sensor_pos_request_scan_elev.get_cached_history()
actual_az = ff.ant2.sensor_pos_actual_scan_azim.get_cached_history()
actual_el = ff.ant2.sensor_pos_actual_scan_elev.get_cached_history()

az_error = np.array(actual_az[1]) - np.array(req_az[1][:len(actual_az[1])])
el_error = np.array(actual_el[1]) - np.array(req_el[1][:len(actual_el[1])])

pylab.plot(actual_az[0], az_error)
pylab.plot(actual_el[0], el_error)
pylab.show()

raw_input("Hit enter to terminate...")
ff.disconnect()
