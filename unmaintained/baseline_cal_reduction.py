#! /usr/bin/python
# Baseline calibration for a single baseline.
#
# Ludwig Schwardt
# 25 January 2010
#

import optparse
import time
import sys

import numpy as np
import matplotlib.pyplot as plt

import scape
import katpoint

# Parse command-line options and arguments
parser = optparse.OptionParser(usage="%prog [options] <data file> [<data file>*]")
parser.add_option('-a', '--baseline', metavar='BASELINE', default='AxAy',
                  help="Baseline to calibrate (e.g. 'A1A2'), default is first interferometric baseline in file")
parser.add_option('-p', '--pol', metavar='POL', default='HH',
                  help="Polarisation term to use ('HH' or 'VV'), default is %default")
parser.add_option('-s', '--max-sigma', type='float', default=0.05,
                  help="Threshold on std deviation of normalised group delay, default is %default")
(opts, args) = parser.parse_args()

if len(args) < 1:
    print 'Please specify the data file to reduce'
    sys.exit(1)

# Load data sets
datasets, scans = [], []
for filename in args:
    print 'Loading baseline', opts.baseline, 'from data file', filename
    d = scape.DataSet(filename, baseline=opts.baseline)
    # Discard 'slew' scans and channels outside the Fringe Finder band
    d = d.select(labelkeep='scan', freqkeep=range(100, 420), copy=True)
    channel_freqs, channel_bw = d.freqs, d.bandwidths[0]
    antenna, antenna2 = d.antenna, d.antenna2
    datasets.append(d)
    scans.extend(d.scans)
time_origin = np.min([scan.timestamps.min() for scan in scans])
# Since the phase slope is sampled, the derived delay exhibits aliasing with a period of 1 / channel_bandwidth
# The maximum delay that can be reliably represented is therefore +- 0.5 / channel_bandwidth
max_delay = 1e-6 / channel_bw / 2
# Maximum standard deviation of delay occurs when delay samples are uniformly distributed between +- max_delay
# In this case, delay is completely random / unknown, and its estimate cannot be trusted
# The division by sqrt(N-1) converts the per-channel standard deviation to a per-snapshot deviation
max_sigma_delay = np.abs(2 * max_delay) / np.sqrt(12) / np.sqrt(len(channel_freqs) - 1)

# Iterate through all scans
group_delay_per_scan, sigma_delay_per_scan, augmented_targetdir_per_scan = [], [], []
raw_delay_per_scan, timestamps, freqs = [], [], []
for scan in scans:
    # Group delay is proportional to phase slope across the band - estimate this as
    # the phase difference between consecutive frequency channels calculated via np.diff.
    # Pick antenna 1 as reference antenna -> correlation product XY* means we
    # actually measure phase(antenna1) - phase(antenna2), therefore flip the sign.
    # Also divide by channel frequency difference, which correctly handles gaps in frequency coverage.
    phase_diff_per_MHz = np.diff(-np.angle(scan.pol(opts.pol)), axis=1) / np.abs(np.diff(channel_freqs))
    # Convert to a delay in seconds
    delay = phase_diff_per_MHz / 1e6 / (-2.0 * np.pi)
    # Raw delay is calculated as an intermediate step for display only - wrap this to primary interval
    raw_delay_per_scan.append(scape.stats.angle_wrap(delay / max_delay, period=1.0).transpose())
    timestamps.append(scan.timestamps - time_origin)
    # Phase differences are associated with frequencies at midpoints between channels
    freqs.append(np.convolve(channel_freqs, [0.5, 0.5], mode='valid'))
    # Obtain robust periodic statistics for *per-channel* phase difference
    delay_stats_mu, delay_stats_sigma = scape.stats.periodic_mu_sigma(delay, axis=1, period=2 * max_delay)
    group_delay_per_scan.append(delay_stats_mu)
    # The estimated mean group delay is the average of N-1 per-channel differences. Since this is less
    # variable than the per-channel data itself, we have to divide the data sigma by sqrt(N-1).
    sigma_delay_per_scan.append(delay_stats_sigma / np.sqrt(len(channel_freqs) - 1))
    # Obtain unit vectors pointing from antenna 1 to target for each timestamp in scan
    az, el = scan.compscan.target.azel(scan.timestamps, antenna)
    targetdir = np.array(katpoint.azel_to_enu(az, el))
    # Invert sign of target vector, as positive dot product with baseline implies negative delay / advance
    # Augment target vector with a 1, as this allows fitting of constant (receiver) delay
    augmented_targetdir_per_scan.append(np.vstack((-targetdir, np.ones(len(el)))))
# Concatenate per-scan arrays into a single array for data set
group_delay = np.hstack(group_delay_per_scan)
sigma_delay = np.hstack(sigma_delay_per_scan)
augmented_targetdir = np.hstack(augmented_targetdir_per_scan)
# Sanitise the uncertainties (can't be too certain...)
sigma_delay[sigma_delay < 1e-5 * max_sigma_delay] = 1e-5 * max_sigma_delay

# Construct design matrix, containing weighted basis functions
A = augmented_targetdir / sigma_delay
# Measurement vector, containing weighted observed delays
b = group_delay / sigma_delay
# Throw out data points with standard deviations above the given threshold
A = A[:, sigma_delay < opts.max_sigma * max_sigma_delay]
b = b[sigma_delay < opts.max_sigma * max_sigma_delay]
print 'Fitting 4 parameters to %d data points (discarded %d)...' % (len(b), len(sigma_delay) - len(b))
# Solve linear least-squares problem using SVD (see NRinC, 2nd ed, Eq. 15.4.17)
U, s, Vt = np.linalg.svd(A.transpose(), full_matrices=False)
augmented_baseline = np.dot(Vt.T, np.dot(U.T, b) / s)
# Also obtain standard errors of parameters (see NRinC, 2nd ed, Eq. 15.4.19)
sigma_augmented_baseline = np.sqrt(np.sum((Vt.T / s[np.newaxis, :]) ** 2, axis=1))

# Convert to useful output (baseline ENU offsets in metres, and receiver delay in seconds)
baseline = augmented_baseline[:3] * katpoint.lightspeed
sigma_baseline = sigma_augmented_baseline[:3] * katpoint.lightspeed
receiver_delay = augmented_baseline[3]
sigma_receiver_delay = sigma_augmented_baseline[3]
# The speed of light in the fibres is lower than in vacuum
cable_lightspeed = katpoint.lightspeed / 1.4
cable_length_diff = augmented_baseline[3] * cable_lightspeed
sigma_cable_length_diff = sigma_augmented_baseline[3] * cable_lightspeed

# First guesses of cable lengths, from cfgdet-array.ini
old_cable_length = {'ant1': 95.5, 'ant2': 108.8, 'ant3': 95.5 + 50, 'ant4': 95.5 + 70, 'ant5': 95.5 + 70, 'ant6': 95.5 + 70, 'ant7': 95.5 + 70}

# Stop the fringes (make a copy of the data first)
scans2 = [scan.select(copy=True) for scan in scans]
fitted_delay_per_scan = []
for n, scan in enumerate(scans2):
    # Store fitted delay and other delays with corresponding timestamps, to allow compacted plot
    fitted_delay = np.dot(augmented_baseline, augmented_targetdir_per_scan[n])
    fitted_delay_per_scan.append(fitted_delay)
    # Stop the fringes (remember that vis phase is antenna1 - antenna2, need to *add* fitted delay to fix it)
    scan.data[:, :, (0 if opts.pol == 'HH' else 1)] *= np.exp(2j * np.pi * np.outer(fitted_delay, channel_freqs * 1e6))
old_baseline = antenna.baseline_toward(antenna2)
old_cable_length_diff = old_cable_length[antenna2.name] - old_cable_length[antenna.name]
old_receiver_delay = old_cable_length_diff / cable_lightspeed
labels = [str(n) for n in xrange(len(scans2))]

# Scale delays to be in units of ADC sample periods, and normalise sigma delay
adc_samplerate = 800e6
for n in xrange(len(scans)):
    group_delay_per_scan[n] *= adc_samplerate
    fitted_delay_per_scan[n] *= adc_samplerate
    sigma_delay_per_scan[n] /= max_sigma_delay

# Produce output plots and results
print "   Baseline (m), old,      stdev"
print "E: %.3f,       %.3f,   %g" % (baseline[0], old_baseline[0], sigma_baseline[0])
print "N: %.3f,       %.3f,   %g" % (baseline[1], old_baseline[1], sigma_baseline[1])
print "U: %.3f,       %.3f,   %g" % (baseline[2], old_baseline[2], sigma_baseline[2])
print "Receiver delay (ns): %.3f, %.3f, %g" % (receiver_delay * 1e9, old_receiver_delay * 1e9,
                                               sigma_receiver_delay * 1e9)
print "Cable length difference (m): %.3f, %.3f, %g" % (cable_length_diff, old_cable_length_diff,
                                                       sigma_cable_length_diff)
print
print "Sources from good to bad"
print "------------------------"
print "(rated on stdev of group delay normalised to max = %g ns):" % (max_sigma_delay * 1e9,)
targets = np.array([scan.compscan.target for scan in scans])
sigma_delays = np.array([sigma_delay.mean() for sigma_delay in sigma_delay_per_scan])
decreasing_performance = np.argsort(sigma_delays)
for target, sigma_delay in zip(targets[decreasing_performance], sigma_delays[decreasing_performance]):
    print "%12s = %.5g" % (target.name, sigma_delay)

plt.figure(1)
plt.clf()
scape.plot_xyz(scans, 'time', 'freq', 'phase')
plt.title('Fringes before stopping')

plt.figure(2)
plt.clf()
scape.plots_basic.plot_segments(timestamps, freqs, raw_delay_per_scan, labels, clim=(-0.5, 0.5))
plt.xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
plt.ylabel('Frequency (MHz)')
plt.title('Raw per-channel delay estimates, as a fraction of maximum delay')

plt.figure(3)
plt.clf()
ax = plt.subplot(211)
scape.plots_basic.plot_segments(timestamps, group_delay_per_scan, labels=labels)
scape.plots_basic.plot_segments(timestamps, fitted_delay_per_scan, color='r')
ax.set_xticklabels([])
ylim_max = np.max(np.abs(ax.get_ylim()))
ax.set_ylim(-max_delay * adc_samplerate, max_delay * adc_samplerate)
plt.ylabel('Delay (ADC samples)')
plt.title('Group delay')
ax = plt.subplot(212)
scape.plots_basic.plot_segments(timestamps, sigma_delay_per_scan)
ax.set_ylim(0, 1)
plt.xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
plt.ylabel('Normalised sigma')
plt.title('Standard deviation of group delay\n(fraction of maximum = %.3g ADC samples)' %
          (max_sigma_delay * adc_samplerate,))
plt.subplots_adjust(hspace=0.25)

plt.figure(4)
plt.clf()
scape.plot_xyz(scans2, 'time', 'freq', 'phase')
plt.title('Fringes after stopping')

# Display plots - this should be called ONLY ONCE, at the VERY END of the script
# The script stops here until you close the plots...
plt.show()
