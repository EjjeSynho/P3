#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:42:49 2021

@author: omartin
"""

# IMPORTING PYTHON LIBRAIRIES
import numpy as np

# IMPORTING P3 MODULES
from aoSystem.telescope import telescope
from aoSystem.atmosphere import atmosphere
from aoSystem.source import source
from aoSystem.deformableMirror import deformableMirror
from aoSystem.detector import detector
from aoSystem.sensor import sensor
from aoSystem.rtc import rtc
import aoSystem.anisoplanatismModel as anisoplanatismModel

#%%
class aoSystem():
    """
    Class initializes AO system parameters with ones given in params dictionary.
    This object must be provided to the PSF models hosted in P3, such as fourierModel,
    psfao21 or psfR.
    """

    def __init__(self, params, getPSDatNGSpositions=False, coo_stars=None, verbose=False):

        self.getPSDatNGSpositions = getPSDatNGSpositions
        self.coo_stars = coo_stars
        self.params = params

        # Telescope
        self.tel = telescope(D               = params['telescope']['TelescopeDiameter'],
                             resolution      = params['telescope']['Resolution'],
                             zenith_angle    = params['telescope']['ZenithAngle'],
                             obsRatio        = params['telescope']['ObscurationRatio'],
                             pupilAngle      = params['telescope']['PupilAngle'],
                             path_pupil      = params['telescope']['PathPupil'],
                             path_static_on  = params['telescope']['PathStaticOn'],
                             path_static_off = params['telescope']['PathStaticOff'],
                             path_static_pos = params['telescope']['PathStaticPos'],
                             path_apodizer   = params['telescope']['PathApodizer'],
                             path_statModes  = params['telescope']['PathStatModes'])
                             
        # Atmosphere
        airmass = 1.0 / np.cos(params['telescope']['ZenithAngle']*np.pi/180.0)
        r0 = 0.976*params['atmosphere']['Wavelength']/params['atmosphere']['Seeing']*3600*180/np.pi

        self.atm = atmosphere(wvl     = params['atmosphere']['Wavelength'],
                              r0      = r0 * airmass**(-3.0/5.0),
                              weights = params['atmosphere']['Cn2Weights'],
                              heights = np.array(params['atmosphere']['Cn2Heights'])*airmass,
                              wSpeed  = params['atmosphere']['WindSpeed'],
                              wDir    = params['atmosphere']['WindDirection'],
                              L0      = params['atmosphere']['L0'])

        # Guide stars
        if params['sources_HO']['Height'] == 0:
            self.ngs = source(wvl     = np.unique(np.array(params['sources_HO']['Wavelength'])),
                              zenith  = params['sources_HO']['Zenith'],
                              azimuth = params['sources_HO']['Azimuth'],
                              tag     = "NGS",
                              verbose = verbose)
            self.lgs = None
        else:
            self.lgs = source(wvl     = np.unique(np.array(params['sources_HO']['Wavelength'])),
                              zenith  = params['sources_HO']['Zenith'],
                              azimuth = params['sources_HO']['Azimuth'],
                              height  = params['sources_HO']['Height']*airmass,
                              tag     = "LGS",
                              verbose = verbose)

            if 'sources_LO' not in params:
                print('(!) WARNING: No information about the tip-tilt star can be retrieved!\n')
                self.ngs = None
            else:
                self.ngs = source(wvl     = np.unique(np.array(params['sources_LO']['Wavelength'])),
                                  zenith  = params['sources_LO']['Zenith'],
                                  azimuth = params['sources_LO']['Azimuth'],
                                  tag     = "NGS",
                                  verbose = verbose)
        # Science sources
        science_zenith  = params['sources_science']['Zenith'] 
        science_azimuth = params['sources_science']['Azimuth']

        if np.any(self.coo_stars):
            science_zenith  = np.hypot(self.coo_stars[0],   self.coo_stars[1])
            science_azimuth = np.arctan2(self.coo_stars[0], self.coo_stars[1])

        # 'Wavelength' in params['sources_LO'])) (?) why wavelength
        if self.getPSDatNGSpositions and 'sources_LO' in params:
            science_zenith  += params['sources_LO']['Zenith']
            science_azimuth += params['sources_LO']['Azimuth']

        self.src = source(wvl     = np.array(params['sources_science']['Wavelength']),
                          zenith  = science_zenith,
                          azimuth = science_azimuth,
                          tag     = "SCIENCE",
                          verbose = verbose)

        # High-order wavefront sensor
        self.wfs = sensor(pixel_scale   = params['sensor_HO']['PixelScale'],
                          fov           = params['sensor_HO']['FieldOfView'],
                          binning       = params['sensor_HO']['Binning'],
                          spotFWHM      = params['sensor_HO']['SpotFWHM'],
                          nph           = params['sensor_HO']['NumberPhotons'],
                          bandwidth     = params['sensor_HO']['SpectralBandwidth'],
                          transmittance = params['sensor_HO']['Transmittance'],
                          dispersion    = params['sensor_HO']['Dispersion'],
                          gain          = params['sensor_HO']['Gain'],
                          ron           = params['sensor_HO']['SigmaRON'],
                          sky           = params['sensor_HO']['SkyBackground'],
                          dark          = params['sensor_HO']['Dark'],
                          excess        = params['sensor_HO']['ExcessNoiseFactor'],
                          nL            = params['sensor_HO']['NumberLenslets'],
                          dsub          = params['sensor_HO']['SizeLenslets'],
                          wfstype       = params['sensor_HO']['WfsType'],
                          modulation    = params['sensor_HO']['Modulation'],
                          noiseVar      = params['sensor_HO']['NoiseVariance'],
                          algorithm     = params['sensor_HO']['Algorithm'],
                          algo_param = [
                              params['sensor_HO']['WindowRadiusWCoG'],
                              params['sensor_HO']['ThresholdWCoG'],
                              params['sensor_HO']['NewValueThrPix']
                              ],
                          tag = "HO WFS")
                          
        if 'sensor_LO' in params:
            # Tip-tilt sensors
            self.tts = sensor(pixel_scale   = params['sensor_LO']['PixelScale'],
                              fov           = params['sensor_LO']['FieldOfView'],
                              binning       = params['sensor_LO']['Binning'],
                              spotFWHM      = params['sensor_LO']['SpotFWHM'],
                              nph           = params['sensor_LO']['NumberPhotons'],
                              bandwidth     = params['sensor_LO']['SpectralBandwidth'],
                              transmittance = params['sensor_LO']['Transmittance'],
                              dispersion    = params['sensor_LO']['Dispersion'],
                              gain          = params['sensor_LO']['Gain'],
                              ron           = params['sensor_LO']['SigmaRON'],
                              sky           = params['sensor_LO']['SkyBackground'],
                              dark          = params['sensor_LO']['Dark'],
                              excess        = params['sensor_LO']['ExcessNoiseFactor'],
                              nL            = params['sensor_LO']['NumberLenslets'],
                              dsub          = params['sensor_LO']['SizeLenslets'],
                              wfstype       = params['sensor_LO']['WfsType'],
                              noiseVar      = params['sensor_LO']['NoiseVariance'],
                              algorithm     = params['sensor_LO']['Algorithm'],
                              algo_param = [
                                  params['sensor_LO']['WindowRadiusWCoG'],
                                  params['sensor_LO']['ThresholdWCoG'],
                                  params['sensor_LO']['NewValueThrPix']
                                  ],
                              tag = "TT WFS")
        else:
            self.tts = None

        #%% Real-time-computer
        self.rtc = rtc(loopGainHO  = params['RTC']['LoopGain_HO'],
                       frameRateHO = params['RTC']['SensorFrameRate_HO'],
                       delayHO     = params['RTC']['LoopDelaySteps_HO'],
                       wfe         = params['RTC']['ResidualError'],
                       loopGainLO  = params['RTC']['LoopGain_LO'],
                       frameRateLO = params['RTC']['SensorFrameRate_LO'],
                       delayLO     = params['RTC']['LoopDelaySteps_LO'])

        # Deformable mirrors
        self.dms = deformableMirror(nActu1D      = params['DM']['NumberActuators'],
                                    pitch        = params['DM']['DmPitchs'],
                                    heights      = params['DM']['DmHeights'],
                                    mechCoupling = params['DM']['InfCoupling'],
                                    modes        = params['DM']['InfModel'],
                                    opt_dir      = [params['DM']['OptimizationZenith'], params['DM']['OptimizationAzimuth']],
                                    opt_weights  = params['DM']['OptimizationWeight'],
                                    opt_cond     = params['DM']['OptimizationConditioning'],
                                    n_rec        = params['DM']['NumberReconstructedLayers'],
                                    AoArea       = params['DM']['AoArea'])
        # Science detector
        self.cam = detector(pixel_scale   = params['sensor_science']['PixelScale'],
                            fov           = params['sensor_science']['FieldOfView'],
                            binning       = params['sensor_science']['Binning'],
                            spotFWHM      = params['sensor_science']['SpotFWHM'],
                            saturation    = params['sensor_science']['Saturation'],
                            nph           = params['sensor_science']['NumberPhotons'],
                            bandwidth     = params['sensor_science']['SpectralBandwidth'],
                            transmittance = params['sensor_science']['Transmittance'],
                            dispersion    = params['sensor_science']['Dispersion'],
                            gain          = params['sensor_science']['Gain'],
                            ron           = params['sensor_science']['SigmaRON'],
                            sky           = params['sensor_science']['SkyBackground'],
                            dark          = params['sensor_science']['Dark'],
                            excess        = params['sensor_science']['ExcessNoiseFactor'],
                            tag           = params['sensor_science']['Name'])
    #%% AO mode
        self.aoMode = 'SCAO'
        
        if self.lgs:
            if self.lgs.nSrc > 1:
                if self.dms.nDMs > 1:
                    self.aoMode = 'MCAO'
                else:
                    if self.dms.nRecLayers >1:
                        self.aoMode = 'LTAO' 
                    else:
                        self.aoMode = 'GLAO'                  
            else:
                self.aoMode = 'SLAO'
    
    #%% ERROR BREAKDOWN
        if self.rtc.holoop['gain'] > 0:
            self.errorBreakdown()
    
    def __repr__(self):
        
        s = '\t\t\t\t________________________'+ self.aoMode + ' SYSTEM ________________________\n\n'
        s += self.src.__repr__() + '\n'
        if self.lgs:
            s += self.lgs.__repr__() + '\n'
        if self.ngs:
            s += self.ngs.__repr__() + '\n'
        s += self.atm.__repr__() + '\n'
        s+= self.tel.__repr__() + '\n'
        s+= self.wfs.__repr__() +'\n'
        if self.tts:
            s+= self.tts.__repr__() +'\n'
        
        s+= self.rtc.__repr__() +'\n'
        s+= self.dms.__repr__() +'\n'
        s+= self.cam.__repr__()

        return s
    
    def errorBreakdown(self):
        """
            Computing the AO wavefront error breakdown based on theoretical formula
        """
        
        rad2nm = lambda x:  np.sqrt(x) * self.atm.wvl*1e9/2/np.pi
        self.wfe = dict()
        
        Dr053 = (self.tel.D/self.atm.r0)**(5/3)
        dactur053 = (self.dms.pitch[0]/self.atm.r0)**(5/3)
        dsubr053 = (self.wfs.optics[0].dsub/self.atm.r0)**(5/3)
        
        # DM fitting error
        self.wfe['DM fitting'] = rad2nm(0.23*dactur053)
        
        # Aliasing error
        self.wfe['WFS aliasing'] = rad2nm(0.07*dsubr053)      
        
        # Servo-lag errors
        ff = np.pi*0.5**2 # to account for the loss of valid actuator outside the pupil
        nMax = int(np.sqrt(ff)*(self.dms.nControlledRadialOrder[0]+1))
        if hasattr(self.rtc,'ttloop') and self.tts !=None :
            nMin = 3
        else:
            nMin = 1
            
        nrad  = np.array(range(nMin,nMax))
        self.wfe['HO Servo-lag'] = rad2nm(0.04 * (self.atm.meanWind/self.tel.D/self.rtc.holoop['bandwidth'])\
                                    * Dr053 * np.sum((nrad+1)**(-2/3)))
        
        # Noise errors        
        if self.wfs.processing.noiseVar == [None]:
            varNoise = self.wfs.NoiseVariance(self.atm.r0 ,self.atm.wvl)
        else:
            varNoise = self.wfs.processing.noiseVar
            
        self.wfe['HO Noise'] = rad2nm(np.mean(varNoise))
        
        if hasattr(self.rtc,'ttloop') and self.tts !=None:
            self.wfe['TT Servo-lag'] = rad2nm(0.04 * (self.atm.meanWind/self.tel.D/self.rtc.ttloop['bandwidth'])* Dr053 * 2**(-2/3))
            
            if self.tts.processing.noiseVar == [None]:
                varNoise = self.tts.NoiseVariance(self.atm.r0 ,self.atm.wvl)
            else:
                varNoise = self.tts.processing.noiseVar
            
            self.wfe['TT Noise']     = rad2nm(np.mean(varNoise))
        else:
            self.wfe['TT Servo-lag'] = 0
            self.wfe['TT Noise'] = 0
        
        # Focal anisoplanatism
        if self.lgs:
            self.wfe['Focal anisoplanatism'] = anisoplanatismModel.focal_anisoplanatism_variance(self.tel,self.atm,self.lgs)
        else:
            self.wfe['Focal anisoplanatism'] = 0
            
        # TO be added : angular anisoplanatisms
           
        self.wfe['Total'] = np.sqrt(self.wfe['DM fitting']**2 + self.wfe['WFS aliasing']**2\
                                    + self.wfe['HO Servo-lag']**2 + self.wfe['HO Noise']**2\
                                    + self.wfe['TT Servo-lag']**2 + self.wfe['TT Noise']**2\
                                    + self.wfe['Focal anisoplanatism']**2)
        self.wfe['Strehl'] = np.exp(-self.wfe['Total']**2 * (2*np.pi*1e-9/self.src.wvl[0])**2)
        