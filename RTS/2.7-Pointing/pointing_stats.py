import sys
import optparse
import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.projections import PolarAxes

import katpoint
from katpoint import rad2deg, deg2rad

def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period



def save_pointingmodel(filebase,model):
    # Save pointing model to file
    outfile = file(filebase + '.csv', 'w')
    outfile.write(model.description)
    outfile.close()
    logger.debug("Saved %d-parameter pointing model to '%s'" % (len(model.params), filebase + '.csv'))



# These fields contain strings, while the rest of the fields are assumed to contain floats
string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']
# Create a date/time string for current time
now = time.strftime('%Y-%m-%d_%Hh%M')


def read_offsetfile(filename):
    # Load data file in one shot as an array of strings
    string_fields = ['dataset', 'target', 'timestamp_ut', 'data_unit']
    data = np.loadtxt(filename, dtype='string', comments='#', delimiter=', ')
    # Interpret first non-comment line as header
    fields = data[0].tolist()
    # By default, all fields are assumed to contain floats
    formats = np.tile(np.float, len(fields))
    # The string_fields are assumed to contain strings - use data's string type, as it is of sufficient length
    formats[[fields.index(name) for name in string_fields if name in fields]] = data.dtype
    # Convert to heterogeneous record array
    data = np.rec.fromarrays(data[1:].transpose(), dtype=zip(fields, formats))
    # Load antenna description string from first line of file and construct antenna object from it
    antenna = katpoint.Antenna(file(filename).readline().strip().partition('=')[2])
    # Use the pointing model contained in antenna object as the old model (if not overridden by file)
    # If the antenna has no model specified, a default null model will be used
    return data


parser = optparse.OptionParser(usage="%prog [options] <data  files > ",
                               description="This fits a pointing model to the given data CSV file"
                               " with the targets that are included in the the offset pointing csv file "
                               " "
                               " ")
parser.add_option( '--offset-file', dest='offset_file', default=None,
                  help="This is the file with the offset test beamfitting results.   "
                  "this is data where a sub section of sources are scaned then the antennas"
                  " Slew away outh to 7 degrees and back to scan the source again")

parser.add_option('-o', '--output', dest='outfilebase', default='pointing_model_%s' % (now,),
                  help="Base name of output files (*.csv for new pointing model and *_data.csv for residuals, "
                  "default is 'pointing_model_<time>')")
parser.add_option('-n', '--no-stats', dest='use_stats', action='store_false', default=True,
                  help="Ignore uncertainties of data points during fitting")
# Minimum pointing uncertainty is arbitrarily set to 1e-12 degrees, which corresponds to a maximum error
# of about 10 nano-arcseconds, as the least-squares solver does not like zero uncertainty
parser.add_option('-m', '--min-rms', type='float', default=np.sqrt(2) * 60. * 1e-12,
                  help="Minimum uncertainty of data points, expressed as the sky RMS in arcminutes")
(opts, args) = parser.parse_args()

if len(args) != 1 or not args[0].endswith('.csv'):
    raise RuntimeError('Please specify a single CSV data file as argument to the script')
filename = args[0]
offset_file = opts.offset_file
min_rms=opts.min_rms
text = []


#offset_file = 'offset_scan.csv'
#filename = '1386710316_point_source_scans.csv'
#min_rms= np.sqrt(2) * 60. * 1e-12

data = read_offsetfile(filename)
keep = np.ones((len(data)),dtype=np.bool)
if not offset_file is None :
    offsetdata = read_offsetfile(offset_file)
    for key,target in enumerate(data['target']):
        keep[key] = target not in set(offsetdata['target'])

# Initialise new pointing model and set default enabled parameters
new_model = katpoint.PointingModel()
num_params = new_model.num_params
default_enabled = np.array([1, 3, 4, 5, 6, 7]) - 1
enabled_params = np.tile(False, num_params)
enabled_params[default_enabled] = True
enabled_params = enabled_params.tolist()



# Fit new pointing model


az, el = angle_wrap(deg2rad(data['azimuth'])), deg2rad(data['elevation'])
measured_delta_az, measured_delta_el = deg2rad(data['delta_azimuth']), deg2rad(data['delta_elevation'])
# Uncertainties are optional
min_std = deg2rad(min_rms  / 60. / np.sqrt(2))
std_delta_az = np.clip(deg2rad(data['delta_azimuth_std']), min_std, np.inf) \
    if 'delta_azimuth_std' in data.dtype.fields and opts.use_stats else np.tile(min_std, len(az))
std_delta_el = np.clip(deg2rad(data['delta_elevation_std']), min_std, np.inf) \
    if 'delta_elevation_std' in data.dtype.fields and opts.use_stats else np.tile(min_std, len(el))

params, sigma_params = new_model.fit(az[keep], el[keep], measured_delta_az[keep], measured_delta_el[keep],
                                     std_delta_az[keep], std_delta_el[keep], enabled_params)


def metrics(model,az,el,measured_delta_az, measured_delta_el ,std_delta_az ,std_delta_el):
    """Determine new residuals and sky RMS from pointing model."""
    model_delta_az, model_delta_el = model.offset(az, el)
    residual_az = measured_delta_az - model_delta_az
    residual_el = measured_delta_el - model_delta_el
    residual_xel  = residual_az * np.cos(el)
    abs_sky_error = rad2deg(np.sqrt(residual_xel ** 2 + residual_el ** 2)) * 60.
    ###### On the calculation of all-sky RMS #####
    # Assume the el and cross-el errors have zero mean, are distributed normally, and are uncorrelated
    # They are therefore described by a 2-dimensional circular Gaussian pdf with zero mean and *per-component*
    # standard deviation of sigma
    # The absolute sky error (== Euclidean length of 2-dim error vector) then has a Rayleigh distribution
    # The RMS sky error has a mean value of sqrt(2) * sigma, since each squared error term is the sum of
    # two squared Gaussian random values, each with an expected value of sigma^2.
    sky_rms = np.sqrt(np.mean(abs_sky_error ** 2))
    # A more robust estimate of the RMS sky error is obtained via the median of the Rayleigh distribution,
    # which is sigma * sqrt(log(4)) -> convert this to the RMS sky error = sqrt(2) * sigma
    robust_sky_rms = np.median(abs_sky_error) * np.sqrt(2. / np.log(4.))
    # The chi^2 value is what is actually optimised by the least-squares fitter (evaluated on the training set)
    chi2 = np.sum(((residual_xel / std_delta_az) ** 2 + (residual_el / std_delta_el) ** 2))
    text = []
    text.append("$\chi^2$ = %.1f " % chi2)
    text.append("all sky rms = %.3f' (robust %.3f') " % (sky_rms, robust_sky_rms))
    return sky_rms,robust_sky_rms,chi2,text


offsetdata

text.append("Blind Pointing metrics for fitted points. (N= %i) "%(np.sum(keep)))
sky_rms,robust_sky_rms,chi2,text1 = metrics(new_model,az[keep],el[keep],measured_delta_az[keep], measured_delta_el[keep] ,std_delta_az[keep] ,std_delta_el[keep])
text += text1
text.append("\n\n")
text.append("Blind Pointing metrics for test points. (Points not used in fit) (N= %i) R.T.P.3"%(np.sum(~keep)))
sky_rms,robust_sky_rms,chi2,text1 = metrics(new_model,az[~keep],el[~keep],measured_delta_az[~keep], measured_delta_el[~keep] ,std_delta_az[~keep] ,std_delta_el[~keep])
text += text1
text.append("\n\n")


for line in text: print line

#print new_model.description

if not offsetdata is None :
    az, el = angle_wrap(deg2rad(offsetdata['azimuth'])), deg2rad(offsetdata['elevation'])
    measured_delta_az, measured_delta_el = deg2rad(offsetdata['delta_azimuth']), deg2rad(offsetdata['delta_elevation'])

    def referencemetrics(measured_delta_az, measured_delta_el):
        """Determine and sky RMS from pointing model."""
        text = []
        measured_delta_xel  =  measured_delta_az* np.cos(el) # scale due to sky shape
        abs_sky_error = np.zeros_like(measured_delta_xel)
        for target in set(offsetdata['target']):
            keep = np.ones((len(offsetdata)),dtype=np.bool)
            for key,targetv in enumerate(offsetdata['target']):
                keep[key] = target == targetv
            abs_sky_error[keep] = rad2deg(np.sqrt((measured_delta_xel[keep]-measured_delta_xel[keep].mean()) ** 2 + (measured_delta_el[keep]- measured_delta_el[keep].mean())** 2)) * 60.
            text.append("%s reference rms = %.3f' (robust %.3f') " % (target,np.sqrt(np.mean(abs_sky_error[keep] ** 2)), np.median(abs_sky_error[keep]) * np.sqrt(2. / np.log(4.))))
        ###### On the calculation of all-sky RMS #####
        # Assume the el and cross-el errors have zero mean, are distributed normally, and are uncorrelated
        # They are therefore described by a 2-dimensional circular Gaussian pdf with zero mean and *per-component*
        # standard deviation of sigma
        # The absolute sky error (== Euclidean length of 2-dim error vector) then has a Rayleigh distribution
        # The RMS sky error has a mean value of sqrt(2) * sigma, since each squared error term is the sum of
        # two squared Gaussian random values, each with an expected value of sigma^2.
        sky_rms = np.sqrt(np.mean(abs_sky_error ** 2))
        print abs_sky_error
        # A more robust estimate of the RMS sky error is obtained via the median of the Rayleigh distribution,
        # which is sigma * sqrt(log(4)) -> convert this to the RMS sky error = sqrt(2) * sigma
        robust_sky_rms = np.median(abs_sky_error) * np.sqrt(2. / np.log(4.))
        text.append("All sky reference rms = %.3f' (robust %.3f') " % (sky_rms, robust_sky_rms))
        return text

    text1 = referencemetrics(measured_delta_az, measured_delta_el)
    text += text1
    print('')
    print(text1)



