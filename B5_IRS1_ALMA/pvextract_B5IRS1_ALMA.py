import numpy as np
from pvextractor import Path, extract_pv_slice, PathFromCenter
from spectral_cube.spectral_cube import SpectralCube
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys
import os
sys.path.append('../B5_NOEMA_30m')
from B5setup import *

lengtharcsec = 15 # in arcsec
widtharcsec = 0.44 # beam major
centerra, centerdec = (ra_yso*u.deg, dec_yso*u.deg) # in degrees
# pvextractor considers 0deg as South to North, we want to go approx. North to South
paangle = (67.1  -90 + 180) * u.deg # degrees, perpendicular to outflow and then N to S instead of S to N
cubename = 'cleaned_images/B5IRS1_C18O_robust05_multi.fits'
cubenameK = 'cleaned_images/B5IRS1_C18O_robust05_multi_K.fits'
pvfilename = 'cleaned_images/B5IRS1_C18O_robust05_multi_K_PV_'+str(paangle)+'deg_'+str(lengtharcsec)+'arcsec.fits'
velinit = 8 * u.km/u.s
velend = 12.5 * u.km/u.s 

if not os.path.exists(cubenameK):
	cube = SpectralCube.read(cubename)
	cube.allow_huge_operations=True
	cube = cube.to(u.K)
	cube.write(cubenameK)
	cube = cube.spectral_slab(velinit, velend)
else:
	cube = SpectralCube.read(cubenameK).spectral_slab(velinit, velend)

gcent = SkyCoord(centerra, centerdec)
pathcent = PathFromCenter(center=gcent, length=lengtharcsec*u.arcsec, angle=paangle, width=widtharcsec*u.arcsec)

slice1 = extract_pv_slice(cube, pathcent)
slice1.writeto(pvfilename)

