"""
This code is meant to be run in python after running prepare_data.py for the first time
Then, go back to CASA and run prepare_data.py again
"""
import astropy.units as u
import os
from spectral_cube import SpectralCube

imagenamebase = "B5-NOEMA+30m-H2CO-1-01-0-00" # was HC3N 10-9
fitsimage = imagenamebase + ".fits"
cutimage = imagenamebase + "_cut.image"
cutfits = imagenamebase + "_cut.fits"

# transform main cube into Kelvin
kelvinfits = imagenamebase + "_cut_K.fits"
if not os.path.exists(kelvinfits):
	imagecube = SpectralCube.read(cutfits).with_spectral_unit(u.km/u.s)
	kelvincube = imagecube.to(u.K)
	kelvincube.hdu.writeto(kelvinfits)
