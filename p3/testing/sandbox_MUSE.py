#%% IMPORTING LIBRARIES
"""
@author: akuznetsov
"""
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
data_dir    = 'C:/Users/akuznets/Data/MUSE/DATA/'
sample_name = 'muse.2021-07-14t23_05_05.869_maoppy.psf.fits'
path_im     = data_dir + sample_name
path_p3     = 'C:/Users/akuznets/repos/TIPTOP/P3'
file_ini    = 'muse_ltao.ini'

date_obs = re.search(r'\d{4}-\d{2}-\d{2}', path_im)[0]

# Read image + variance cubes
cube_img = []
cube_var = []
wavelengths = []

with fits.open(path_im) as hdul:
    obs_info = hdul[0].header
    for bandwidth_id in range(0,10):
        cube_img.append(hdul[bandwidth_id*4+1].data)
        cube_var.append(hdul[bandwidth_id*4+2].data)
        wavelengths.append(hdul[bandwidth_id*4+1].header['LAMBDAOB']*1e-10)

angle = -44

# In this case, the image is rotate instead of the pupil
for i in range(len(cube_img)):
    cube_img[i] = rotate(cube_img[i], angle)
    cube_var[i] = rotate(cube_var[i], angle)

wavelengths = np.array(wavelengths)
cube_img    = np.dstack(cube_img)
cube_var    = np.dstack(cube_var)
polychrome  = cube_img.sum(axis=2)

#Find maximum of the composite PSF
crop_size = 200
ind = np.unravel_index(np.argmax(polychrome, axis=None), polychrome.shape) 
win_x = [ ind[0]-crop_size//2, ind[0]+crop_size//2 ]
win_y = [ ind[1]-crop_size//2, ind[1]+crop_size//2 ]

#Crop image cubes
cube_img   = cube_img  [ win_x[0]:win_x[1], win_y[0]:win_y[1], :]
cube_var   = cube_var  [ win_x[0]:win_x[1], win_y[0]:win_y[1], :]
polychrome = polychrome[ win_x[0]:win_x[1], win_y[0]:win_y[1]]

wvl_id = 5

wvl = wavelengths[wvl_id]
im  = cube_img[:,:,wvl_id]
var = cube_var[:,:,wvl_id]

im  = im[1:,1:]
var = var[1:,1:]

parser = parameterParser(file_ini)
params = parser.params

OB_NAME = obs_info['TARGNAM']
RA      = float(obs_info['RA'])
DEC     = float(obs_info['DEC'])

# query simbad
DICT_SIMBAD = query_simbad(Time(obs_info['DATE-OBS']), SkyCoord(RA*u.degree,DEC*u.degree), name=OB_NAME)

# get magnitudes
if 'simbad_FLUX_V' in DICT_SIMBAD: VMAG = DICT_SIMBAD['simbad_FLUX_V']
if 'simbad_FLUX_R' in DICT_SIMBAD: RMAG = DICT_SIMBAD['simbad_FLUX_R']
if 'simbad_FLUX_G' in DICT_SIMBAD: GMAG = DICT_SIMBAD['simbad_FLUX_G']
if 'simbad_FLUX_J' in DICT_SIMBAD: JMAG = DICT_SIMBAD['simbad_FLUX_J']
if 'simbad_FLUX_H' in DICT_SIMBAD: HMAG = DICT_SIMBAD['simbad_FLUX_H']
if 'simbad_FLUX_K' in DICT_SIMBAD: KMAG = DICT_SIMBAD['simbad_FLUX_K']

airmass = obs_info['AIRMASS']

params['atmosphere']['Seeing'] = obs_info['SPTSEEIN']
params['atmosphere']['L0'] = obs_info['SPTL0']
params['atmosphere']['WindSpeed'] = [obs_info['WINDSP']] * len(params['atmosphere']['Cn2Heights'])
params['atmosphere']['WindDirection'] = [obs_info['WINDIR']] * len(params['atmosphere']['Cn2Heights'])

params['sources_science']['Wavelength'] = [wvl]
params['sensor_science']['FieldOfView'] = crop_size
params['sensor_science']['Zenith']  = [90.0-obs_info['TELALT']]
params['sensor_science']['Azimuth'] = [obs_info['TELAZ']]

params['sensor_HO']['NoiseVariance'] = 5.0
rad2mas = 3600 * 180 * 1e3 / np.pi

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

x0 = {
    'sensor_HO.NoiseVariance': params['sensor_HO']['NoiseVariance'],
    'atmosphere.Seeing': params['atmosphere']['Seeing'],
    'jitterX': fao.ao.cam.spotFWHM[0][0],
    'jitterY': fao.ao.cam.spotFWHM[0][1],
    'Jxy':     fao.ao.cam.spotFWHM[0][2],
    'F':       1.0,
    'dx':      0.,
    'dy':      0.,
    'bkg':     0.
}

# Let's optimize them all at once and let them happily cross-entangle
fixed = {
    'sensor_HO.NoiseVariance': False,
    'atmosphere.Seeing': False,
    'jitterX': False,
    'jitterY': False,
    'Jxy':     True,
    'F':       False,
    'dx':      False,
    'dy':      False,
    'bkg':     False
}

res_Fourier_0 = psfFitting(im, fao, x0, fixed, verbose=2, normType=1)
x1 = deepcopy(res_Fourier_0.x)
displayResults(fao, res_Fourier_0, vmin=20, vminc=40, vmaxc=np.max(res_Fourier_0.im_sky)*1.1, nBox=140, scale='log10abs', y_label=r'Intensity, $[F_{\lambda}/10^{-20}ergs...etc]$', show=False)

# %%
