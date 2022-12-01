'''
This file uses licpy to work, which requires tensorflow 1.14 (not tensorflow 2.0 at the time of writing).
Please make sure you are in an environment where you can use tensorflow. The package cannot be
downloaded if regions==0.5 or radio-beam==0.3.2 are installed (at least through pip)

The only part that uses licpy is for the visualization. The rest should have no issues
'''

import numpy as np
import os
import scipy.linalg # we only fit a linear plane so we go with easy linear code
from astropy.io import fits
import astropy.units as u
import sys
sys.path.append('../')
from B5setup import *
import matplotlib.pyplot as plt
from astropy.wcs import WCS

gaussfitfolder = 'gaussfit/'
velfile = gaussfitfolder + 'B5-NOEMA+30m-H3CN-10-9_cut_K_1G_fitparams_filtered_Vlsr'
Tpeakfile = gaussfitfolder + 'B5-NOEMA+30m-H3CN-10-9_cut_K_1G_fitparams_filtered_Tpeak'
nablavelfile = velfile + '_gradient'
epsilon = 1e-3 # tolerance to mask values
# values for LIC
nbeams = 8

# we need to sample an area of about 2 beams in principle
# we can try for different sample radii
# this will sample gradients along larger structures
velfield, velheader = fits.getdata(velfile+'.fits', header=True)
wcsvel = WCS(velheader)
y_velfield, x_velfield = np.mgrid[:len(velfield), :len(velfield[0])]
beammaj, beammin = velheader['BMAJ'], velheader['BMIN']
pixsize_deg = np.abs(velheader['CDELT2'])
pixsize_pc = (pixsize_deg * 3600 * dist_B5.value * u.au).to(u.pc).value # this is pixel to pc
equivradius = np.sqrt(beammaj * beammin / (4 * np.log(2)))
sampleradius = int(np.round(nbeams* equivradius / pixsize_deg, 0))
beamarea_pix = np.pi * beammaj * beammin / (4 * np.log(2) * pixsize_deg**2)

if not os.path.exists('gradient_x_{0}beams.fits'.format(nbeams)) or not os.path.exists('gradient_y_{0}beams.fits'.format(nbeams)):
	filled_indices = np.where(~np.isnan(velfield))
	nablax_map = np.ones(np.shape(velfield)) * np.nan
	nablay_map = np.ones(np.shape(velfield)) * np.nan
	absnabla_map = np.ones(np.shape(velfield)) * np.nan
	
	for y, x in np.transpose(filled_indices):
		sampleregion = velfield[y-sampleradius:y+sampleradius+1, x-sampleradius:x+sampleradius+1]
		if np.shape(sampleregion)[0] != np.shape(sampleregion)[1]: 
			continue
		elif np.sum(~np.isnan(sampleregion))<beamarea_pix:
			continue
		X,Y = np.meshgrid(np.arange(-sampleradius, sampleradius+1, 1), np.arange(-sampleradius, sampleradius+1, 1))
		index_sampleregion_filter = np.where(distancepix(X, Y, sampleradius+1, sampleradius+1)<sampleradius * ~np.isnan(sampleregion))
		# we need to get the gradient in km/s/pc
		data_sample = np.transpose([X[index_sampleregion_filter]*pixsize_pc, Y[index_sampleregion_filter]*pixsize_pc, sampleregion[index_sampleregion_filter]])
		# data_sample_filtered = data_sample[:, ~np.isnan(data_sample[:, 2])]
		A = np.c_[data_sample[:,0],data_sample[:,1], np.ones(data_sample.shape[0])]
		C,_,_,_ = scipy.linalg.lstsq(A, data_sample[:,2])
		if np.abs(C[0])< epsilon or np.abs(C[1])< epsilon:
			continue
		nablax_map[y, x] = C[0]
		nablay_map[y, x] = C[1]
		absnabla_map[y, x] = np.sqrt(C[1]**2 + C[0]**2)
	
	gradheader = velheader.copy()
	
	gradheader['BUNIT'] = 'km s-1 pc-1'
	fits.writeto('gradient_x_{0}beams.fits'.format(nbeams), nablax_map, gradheader, overwrite=True)
	fits.writeto('gradient_y_{0}beams.fits'.format(nbeams), nablay_map, gradheader, overwrite=True)
	fits.writeto('gradient_abs_{0}beams.fits'.format(nbeams), absnabla_map, gradheader, overwrite=True)
	
else:
	nablax_map, gradheader = fits.getdata('gradient_x_{0}beams.fits'.format(nbeams), header=True)
	nablay_map = fits.getdata('gradient_y_{0}beams.fits'.format(nbeams))
	absnabla_map = fits.getdata('gradient_abs_{0}beams.fits'.format(nbeams))

# LIC visualization
from licpy.lic import runlic

L = [4,8,10,20] #pix
for Li in L:
	dest = 'gradient_LIC_{0}beams_{1}_absnabla'.format(nbeams, Li)
	tex = runlic(nablax_map, nablay_map, Li)
	
	fig = plt.figure(figsize=(6,6))
	ax = fig.add_subplot(111, projection=wcsvel)
	ax.imshow(tex * absnabla_map, vmin = 0, vmax = 30, origin='lower')
	fig.savefig(dest+'.pdf', bbox_inches='tight', dpi=100)
	dest2 = 'gradient_LIC_{0}beams_{1}'.format(nbeams, Li)
	fig2 = plt.figure(figsize=(6,6))
	ax2 = fig2.add_subplot(111)
	ax2.imshow(tex, cmap='Greys', origin='lower')
	fig2.savefig(dest2+'.pdf', bbox_inches='tight', dpi=100)
	
	fits.writeto(dest2+'.fits', tex, gradheader)
