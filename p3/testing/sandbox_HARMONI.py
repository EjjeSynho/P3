#%% IMPORTING LIBRARIES
"""
@author: akuznetsov
"""

%reload_ext autoreload
%autoreload 2

import sys
import os

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

from aoSystem.fourierModel import fourierModel
from aoSystem.aoSystem import aoSystem
from copy import deepcopy

from psfFitting.psfFitting import psfFitting
from psfFitting.psfFitting import displayResults

from aoSystem.parameterParser import parameterParser
from query_eso_archive import query_simbad

from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.ndimage import rotate


#%% ------------------------ Managing paths ------------------------
path_p3     = 'C:/Users/akuznets/repos/TIPTOP/P3/p3/'
file_ini    = 'HarmoniLTAO_3.ini'


parser = parameterParser(file_ini)
params = parser.params

#%% -------------- Fitting a monochromatic slice --------------
# Which PSD contributors to include into the model
include_psd = {
    'noise':           True,
    'aliasing':        True,
    'diff_refr':       True,
    'chromatism':      True,
    'spatio_temporal': True,
    'fitting':         True,
    'only_fitting':    False
}

ao  = aoSystem(params, getPSDatNGSpositions=False, verbose=False)
fao = fourierModel(ao,  calcPSF=False, display=False, verbose=False, include_psd=include_psd) #instantiating the model

#%%
PSD = {
    'Noise' :          fao.psdNoise * fao.psdNormFactor,
    'Alias' :          fao.psdAlias * fao.psdNormFactor,
    'DiffRef' :        fao.psdDiffRef * fao.psdNormFactor,
    'Chromatism' :     fao.psdChromatism * fao.psdNormFactor,
    'SpatioTemporal' : fao.psdSpatioTemporal * fao.psdNormFactor,
    'Fit' :            fao.psdFit * fao.psdNormFactor,
    'Total' :          fao.psd * fao.psdNormFactor
}
plt.imshow(np.abs(PSD['Total']))



# %%
import numpy as np
from astropy.io import fits

# Sample Python dictionary with 2D matrices of floats
data_dict = PSD
# Create an empty list to store the ImageHDUs
image_hdus = []

# Iterate over the dictionary to create ImageHDUs
for key, matrix in data_dict.items():
    # Create a new ImageHDU with the 2D matrix data
    image_hdu = fits.ImageHDU(data=matrix.squeeze())
    
    # Set the 'EXTNAME' header keyword to store the key
    image_hdu.header.set('EXTNAME', key)

    # Append the ImageHDU to the list
    image_hdus.append(image_hdu)

# Create a primary HDU with no data, only header
primary_hdu = fits.PrimaryHDU()

# Create the HDU list and write the FITS file
hdul = fits.HDUList([primary_hdu] + image_hdus)
hdul.writeto('output.fits', overwrite=True)

#%%

# Read fits file
with fits.open('output.fits') as HDUL:
    # for hdu in HDUL:
    #     print(hdu.header)
    #     print(hdu.data)
    HDUL.info()
# %%
