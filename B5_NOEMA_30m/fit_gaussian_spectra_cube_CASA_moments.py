"""
This file was the first run to obtain a Gaussian fit for HC3N in B5. It uses the moments calculated in Jy/beam,
Which is why in the code they are transformed to Kelvin. The moments saved in the folder casamomentstest/ are
the ones used here, which have units of Jy/beam

"""
from spectral_cube import SpectralCube
import pyspeckit
import matplotlib.pylab as plt
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
import os
from astropy.modeling.functional_models import Gaussian2D
from astropy.coordinates import SkyCoord
from regions import Regions


cubefile = 'casamomentstest/B5-NOEMA+30m-H3CN-10-9_cut'
# Where we estimate the line is
velinit = 9.0 * u.km/u.s
velend = 11.3 * u.km/u.s
starting_point = (107,166) #x, y
signal_cut = 3
snratio = 3
# We already calculated the rms through CASA in the channels without emission
rms = 22.84 * u.mJy/u.beam

if not os.path.exists(cubefile+'_K.fits'):
    # For B5, the cube must be in K but we fit all the map
    cube1 = SpectralCube.read(cubefile+'.fits').with_spectral_unit(u.km/u.s)
    header1 = cube1.header
    restfreq = header1['restfrq'] * u.Hz
    beamarea = 1.133 * header1['bmaj'] * header1['bmin'] * u.deg**2
    cube1 = cube1.to(u.K) #, u.brightness_temperature(restfreq, beam_area=beamarea))
    cube1.hdu.writeto(cubefile+'_K.fits')

spc = pyspeckit.Cube(cubefile+'_K.fits')
header = spc.header
ra = header['ra'] #phasecent
dec = header['dec']
naxis = header['naxis1']
freq = (header['RESTFRQ']/1e9) * u.GHz
wcsspec = WCS(header).spectral
wcscel = WCS(header).celestial
beamarea = 1.133 * header['bmaj'] * header['bmin'] * u.deg**2
rms_K = rms.to(u.K, u.brightness_temperature(freq, beam_area=beamarea))
rmsmap = np.ones(np.shape(spc.cube)) * rms_K.value
#spc.errorcube = rms_K.value
# chanlims = [wcsspec.world_to_pixel(velinit).tolist(), wcsspec.world_to_pixel(velend).tolist()]
#rmsmap = np.ones(np.shape(spc.cube)) * rms


def filter(spc, rms, rmslevel, velinit, velend, negative=True, errorfrac=10, epsilon=1.e-6, region=None):
    """
    Replace the pixels in the fitted cube with np.nan where the fit is not
    good enough according to our criteria.

    The criteria that a pixel must have are:
    - The error is not zero
    - The value must not be negative (in this case we know the moment 1 must be
    positive, so we specify negative=True, can be changed)
    - The error fraction is lower than errorfrac, not applied if errorfrac>=1
    - The moment 1 value must be within the range [velinit,velend]
    - The peak value must be larger than rms times rmslevel
    - The weighted velocity dispersion must be smaller than the absolute
    value of velend-velinit
    - If one parameter in a spectra is np.nan, all the spectra must be nan (sanity
    check)
    - All points must be within a region (part of the input)

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    zeromask = np.where(np.abs(spc.errcube[0]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[1]) < epsilon, 1, 0) + \
        np.where(np.abs(spc.errcube[2]) < epsilon, 1, 0)
    spc.parcube[np.where(np.repeat([zeromask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([zeromask], 3, axis=0))] = np.nan

    if errorfrac < 1.0:
    	errormask = np.where(np.abs(spc.errcube[0]/spc.parcube[0]) > errorfrac, 1, 0)\
        	+ np.where(np.abs(spc.errcube[1]/spc.parcube[1]) > errorfrac, 1, 0)\
        	+ np.where(np.abs(spc.errcube[1]/spc.parcube[1]) > errorfrac, 1, 0)
    	spc.parcube[np.where(np.repeat([errormask], 3, axis=0))] = np.nan
    	spc.errcube[np.where(np.repeat([errormask], 3, axis=0))] = np.nan


    if negative:
        negativemask = np.where(spc.parcube[0] < 0, 1, 0) + \
            np.where(spc.parcube[1] < 0, 1, 0) + \
            np.where(spc.parcube[2] < 0, 1, 0)
        spc.parcube[np.where(np.repeat([negativemask], 3, axis=0))] = np.nan
        spc.errcube[np.where(np.repeat([negativemask], 3, axis=0))] = np.nan

    velocitymask = np.where(spc.parcube[1] < velinit.value, 1, 0) + \
        np.where(spc.parcube[1] > velend.value, 1, 0)
    spc.parcube[np.where(np.repeat([velocitymask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([velocitymask], 3, axis=0))] = np.nan

    sigmamask = np.where(spc.parcube[2] <= np.abs(header['cdelt3']/1000), 1, 0) # CDELT3 is in m/s
    spc.parcube[np.where(np.repeat([sigmamask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([sigmamask], 3, axis=0))] = np.nan

    peakmask = np.where(spc.parcube[0] < rmslevel*rms, 1, 0)
    spc.parcube[np.where(np.repeat([peakmask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([peakmask], 3, axis=0))] = np.nan

    # Force if one parameter in a channel is nan, all the same pixels
    # in all  channels must be nan
    nanmask = np.sum(np.where(np.isnan(np.concatenate([spc.parcube, spc.errcube])), 1, 0), axis=0)
    spc.parcube[np.where(np.repeat([nanmask], 3, axis=0))] = np.nan
    spc.errcube[np.where(np.repeat([nanmask], 3, axis=0))] = np.nan

    if region is not None:
        cutout = Regions.read(region)[0].to_pixel(wcscel)
        cutoutmask = 1 - cutout.to_mask().to_image(np.shape(spc.parcube)[1:])
        spc.parcube[np.where(np.repeat([cutoutmask], 3, axis=0))] = np.nan
        spc.errcube[np.where(np.repeat([cutoutmask], 3, axis=0))] = np.nan

    return spc


# as moments we will use the moments created by CASA
# remember to transform to K

mom8 = fits.getdata(cubefile + '_mom8.fits') * u.Jy/u.beam 
mom1 = fits.getdata(cubefile + '_mom1.fits') # * u.km/u.s
mom2 = fits.getdata(cubefile + '_mom2.fits') # * u.km/u.s

mom8_K = mom8.to(u.K, u.brightness_temperature(freq, beam_area=beamarea))
mom8_K[np.where(mom8.value < rms_K.value*snratio)] = rms_K*snratio

mom8_init = mom8_K.value
mom2_init = np.where(mom2>0.01, mom2, 0.01)
mom1[np.where(mom1<velinit.value, 1, 0) * np.where(mom1>velend.value, 1, 0)] = 10.0

# mom01 = np.where(spc.momentcube[0]>rms_K.value * snratio, spc.momentcube[0], rms_K.value * snratio)
# initguesses = np.array([mom01,spc.momentcube[1],spc.momentcube[2]])
initguesses = np.array([mom8_init,mom1,mom2_init])
fitfile = cubefile + '_K_1G_fitparams.fits'
fitfile_filtered = cubefile + '_K_1G_fitparams_filtered.fits'
regionmask_name = 'casamomentstest/selection_region_mask.reg'

if os.path.exists(fitfile):
    spc.load_model_fit(fitfile, 3, fittype='gaussian')
    spc = filter(spc, rms_K.value, snratio, velinit, velend, errorfrac=0.5, region=regionmask_name)
    spc.write_fit(fitfile_filtered, overwrite=True)
    fittedmodel_filtered = spc.get_modelcube()
else:
    try:
        spc.fiteach(fittype='gaussian',
                    guesses=initguesses,
		    use_neighbor_as_guess=False,
                    verbose=1,
		    negamp=False,
                    errmap=rmsmap,
                    signal_cut=signal_cut,
                    blank_value=np.nan,
                    start_from_point=(starting_point))
    except AssertionError:
        print('There are non-finite parameters in the fit')
    #spc = filter(spc, rms_K.value, snratio, velinit, velend)
    spc.write_fit(fitfile)
    fittedmodel = spc.get_modelcube()

tmax, vlsr, sigmav = spc.parcube
key_list = ['NAXIS3', 'CRPIX3', 'CDELT3', 'CUNIT3', 'CTYPE3', 'CRVAL3']

commonhead = fits.getheader(fitfile)
for key_i in key_list:
    commonhead.remove(key_i)
commonhead['NAXIS'] = 2
commonhead['WCSAXES'] = 2
headervelocities = commonhead.copy()
headervelocities['BUNIT'] = 'km/s'
if not os.path.exists(cubefile + '_K_1G_tmax.fits'):
    hdutmax = fits.PrimaryHDU(data=tmax, header=commonhead)
    hdutmax.writeto(cubefile + '_K_1G_tmax.fits', overwrite=True)
if not os.path.exists(cubefile + '_K_1G_Vc.fits'):
    hduvlsr = fits.PrimaryHDU(data=vlsr, header=headervelocities)
    hduvlsr.writeto(cubefile + '_K_1G_Vc.fits', overwrite=True)
if not os.path.exists(cubefile + '_K_1G_sigma_v.fits'):
    hdusigmav = fits.PrimaryHDU(data=sigmav, header=headervelocities)
    hdusigmav.writeto(cubefile + '_K_1G_sigma_v.fits', overwrite=True)
if not os.path.exists(cubefile + 'K_fitted.fits'):
    modelhdu = fits.PrimaryHDU(data=fittedmodel, header=header)
    modelhdu.writeto(cubefile + 'K_fitted.fits', overwrite=True)
if not os.path.exists(cubefile + 'K_fitted_filtered.fits'):
    modelhdu = fits.PrimaryHDU(data=fittedmodel_filtered, header=header)
    modelhdu.writeto(cubefile + 'K_fitted.fits', overwrite=True)
