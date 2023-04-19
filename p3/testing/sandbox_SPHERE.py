#%%
"""
@author: akuznetsov
"""
# Importing libraries
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import copy
from copy import deepcopy

from aoSystem.fourierModel import fourierModel
from aoSystem.aoSystem import aoSystem
from aoSystem.parameterParser import parameterParser

from psfFitting.psfFitting import psfFitting
from psfFitting.psfFitting import displayResults

import telemetry.sphereUtils as sphereUtils
import aoSystem.FourierUtils as FourierUtils
import re

from os import path
import pickle 

#%%===================================================================
path_root = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
path_ini = path.join(path_root, path.normpath('aoSystem/parFiles/irdis.ini'))

path_test = path.join(
    path_root, \
    path.normpath('testing/735_SPHER.2017-05-15T04.48.44.108IRD_FLUX_CALIB_CORO_RAW_left.pickle'))
file_ini  = 'irdis.ini'

with open(path_test, 'rb') as handle:
    data_test = pickle.load(handle)

parser = parameterParser(file_ini)
params = parser.params

im = data_test['image']
wvl = data_test['spectrum']['lambda']

params['atmosphere']['Seeing'] = data_test['seeing']['SPARTA']
params['atmosphere']['WindSpeed'] = [data_test['Wind speed']['header']]
params['atmosphere']['WindDirection'] = [data_test['Wind direction']['header']]
params['sources_science']['Wavelength'] = [wvl]
params['sensor_science']['FieldOfView'] = data_test['image'].shape[0]
params['sensor_science']['Zenith'] = [90.0-data_test['telescope']['altitude']]
params['sensor_science']['Azimuth'] = [data_test['telescope']['azimuth']]
params['sensor_science']['SigmaRON'] = data_test['Detector']['ron']
params['sensor_science']['Gain'] = data_test['Detector']['gain']

params['RTC']['SensorFrameRate_HO']  = data_test['WFS']['rate']
params['sensor_HO']['NumberPhotons'] = [data_test['WFS']['Nph vis']]

seeing = params['atmosphere']['Seeing']

#%% ------------------------ Init fitting ------------------------

include_psd = {
    'noise':           True,
    'aliasing':        True,
    'diff_refr':       True,
    'chromatism':      True,
    'spatio_temporal': True,
    'fitting':         True,
    'only_fitting':    False
}

ao  = aoSystem(params, getPSDatNGSpositions=False, verbose=True)
fao = fourierModel(ao, calcPSF=False, display=False, verbose=True, include_psd=include_psd) #instantiating the model

x0 = {
    'sensor_HO.NoiseVariance': 0.1,
    'atmosphere.Seeing': seeing,
    'jitterX': fao.ao.cam.spotFWHM[0][0],
    'jitterY': fao.ao.cam.spotFWHM[0][1],
    'Jxy':     fao.ao.cam.spotFWHM[0][2],
    'F':       1.,
    'dx':      0.,
    'dy':      0.,
    'bkg':     0.
}

fixed = {
    'sensor_HO.NoiseVariance': True,
    'atmosphere.Seeing': True,
    'jitterX': False,
    'jitterY': False,
    'Jxy':     True,
    'F':       False,
    'dx':      False,
    'dy':      False,
    'bkg':     False,
}

# fitting the residual jitter + astrometry/photometry
res_Fourier = psfFitting(im, fao, x0, fixed, verbose=2, normType=1)
displayResults(fao,res_Fourier,nBox=90,scale='log10abs', y_label='Intensity, [ADUs]')
# %%
