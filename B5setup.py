import aplpy
import astropy.units as u
import sys
import numpy as np
from scipy import stats
sys.path.append('/home/mvaldivi/velocity_tools')
# definition of constants and fuctions

ra_yso = (3 +(47+41.591/60)/60)*15
dec_yso = 32+(51+43.672/60)/60 # Tobin et al 2016 catalog
dist_B5 = 302 * u.pc # pc, Zucker et al 2018?

def plot_aplpy_subfig(image, figure, subplotindex, stretch, vmin, vmax, cmap, vmid=0.01, label_col='k', distance=302., barsize=5000, showframeScalebar=False):
    """
    This code works for both single plot figures (subplotindex=(1,1,1)) and for subplots
    
    Args:
        image (ImageHDU): either the name of the file in .fits containing the image to plot
                            or an HDU
        figure (matplotlib.figure.Figure): figure where to put the image
        subplotindex (tuple): 3 element array with nrows, ncols, and index where to put the
                                image
        stretch (string): name of the stretch for the colorbar. Can be 'linear', 'log', 'arcsinh', etc
        vmin, vmax (float): minimum and maximum values to plot in the colorbar
        cmap (string): name of the matplotlib colormap to use for the image
    
    Returns: 
        aplpy.FITSFigure: aplpy instance which is placed in the figure
    """
    
    fig = aplpy.FITSFigure(image, figure=figure, subplot=subplotindex)
    if stretch=='arcsinh':
        fig.show_colorscale(stretch=stretch, cmap=cmap, vmin=vmin, vmax=vmax, vmid=vmid)
    else:
        fig.show_colorscale(stretch=stretch, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.add_colorbar()
    # set properties of the axis
    fig.set_system_latex(True)
    # fig.ticks.set_color(label_col)
    fig.tick_labels.set_xformat('hh:mm:ss')
    fig.tick_labels.set_yformat('dd:mm:ss')
    fig.axis_labels.set_xtext(r'Right Ascension (J2000)')
    fig.axis_labels.set_ytext(r'Declination (J2000)')
    # set beam
    fig.add_beam(color=label_col)
    fig.beam.set_color('k')
    #set scalebar
    ang_size = (barsize / distance) * u.arcsec
    fig.add_scalebar(ang_size, label=str(barsize)+' au', color=label_col, corner='top right', frame=showframeScalebar)
    fig.scalebar.set_linewidth(2)
    #set bad color
    fig.set_nan_color('0.85')
    
    return fig

def get_vc_r(velfield_file, centerra, centerdec, distance, region_file=None):
	
	"""
    Returns the centroid velocity and projected separation in the sky for the
    centroid velocity from Per-emb-50

    Given a region and a velocity field for the vicinity of a protostar,
    obtains the projected radius and the central velocity of each pixel in the
    region. The velocity field must be masked to contain only the relevant
    pixels.

    Args:
        velfield_file (string): path to the .fits file containing the velocity
        field
        region_file (string): path to the .reg (ds9) region file where the
        streamer is contained
        centerra, centerdec (Units: deg): central position of the protostar in degrees (J2000)
        distance (Units: pc): distance to the protostar in pc

    Returns:
        type: description

    """
	from regions import Regions
	from astropy.wcs import WCS
	from astropy.io import fits
	import velocity_tools.coordinate_offsets as c_offset
	# load region file and WCS structures
	if region_file is not None: regions = Regions.read(region_file)
	wcs_Vc = WCS(velfield_file)
	hd_Vc = fits.getheader(velfield_file)
	results = c_offset.generate_offsets(hd_Vc, centerra, centerdec, pa_angle=0*u.deg, inclination=0*u.deg)
	rad_au = (results.r * distance*u.pc).to(u.au, equivalencies=u.dimensionless_angles())
	if region_file is not None:
		mask_Vc = (regions[0].to_pixel(wcs_Vc)).to_mask()
		Vc_cutout = mask_Vc.cutout(fits.getdata(velfield_file))
		rad_cutout = mask_Vc.cutout(rad_au)
		gd = (mask_Vc.data == 1)
	else:
		Vc_cutout = fits.getdata(velfield_file)
		gd = ~np.isnan(Vc_cutout)
		rad_cutout = rad_au
	v_los = Vc_cutout[gd]*u.km/u.s
	r_proj = rad_cutout[gd]
	return r_proj, v_los



def primary_beam_alma(freq, diameter=12):
	freq0 = 300 * u.GHz
	fwhm0_12 = 21 * u.arcsec
	fwhm0_7 = 35 * u.arcsec
	if diameter==12:
		return fwhm0_12 / freq0 * freq
	elif diameter==7:
		return fwhm0_7 / freq0 * freq
	else:
		print('Please select either 12 or 7 to calculate the primary beam FWHM for a 12m or 7m antenna')
		return