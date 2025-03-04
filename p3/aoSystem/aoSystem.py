#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:42:49 2021

@author: omartin
"""

# IMPORTING PYTHON LIBRAIRIES
import os.path as ospath
from configparser import ConfigParser
import numpy as np

import yaml

# IMPORTING P3 MODULES
from p3.aoSystem.telescope import telescope
from p3.aoSystem.atmosphere import atmosphere
from p3.aoSystem.source import source
from p3.aoSystem.deformableMirror import deformableMirror
from p3.aoSystem.detector import detector
from p3.aoSystem.sensor import sensor
from p3.aoSystem.rtc import rtc
import p3.aoSystem.anisoplanatismModel as anisoplanatismModel

#%%
class aoSystem():

    

    def check_section_key(self, primary):
        if self.configType == 'ini':
            return self.config.has_section(primary)
        elif self.configType == 'yml':
            return primary in self.my_yaml_dict.keys()


    
    def check_config_key(self, primary, secondary):
        if self.configType == 'ini':
            return self.config.has_option(primary, secondary)
        elif self.configType == 'yml':
            if primary in self.my_yaml_dict.keys():
                return secondary in self.my_yaml_dict[primary].keys()
            else:
                return False


    def get_config_value(self, primary, secondary):
        if self.configType == 'ini':
            return eval(self.config[primary][secondary])
        elif self.configType == 'yml':
            return self.my_yaml_dict[primary][secondary]

    def __init__(self,path_config,path_root='',nLayer=None,getPSDatNGSpositions=False,coo_stars=None):
                            
        self.error = False
        # verify if the file exists
        if ospath.isfile(path_config) == False:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The .ini or .yml file does not exist\n')
            self.error = True
            return
                
        if path_config[-4::]=='.ini':
            # open the .ini file
            self.configType = 'ini'
            self.config = ConfigParser()
            self.config.optionxform = str
            self.config.read(path_config)

        elif path_config[-4::]=='.yml':
            self.configType = 'yml'
            with open(path_config) as f:
                self.my_yaml_dict = yaml.safe_load(f)
        
        self.getPSDatNGSpositions = getPSDatNGSpositions
        
        #%% TELESCOPE
        #----- grabbing main parameters
        if self.check_config_key('telescope','TelescopeDiameter'):
            D = self.get_config_value('telescope','TelescopeDiameter')
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the telescope diameter\n')
            self.error = True
            return
        
        if self.check_config_key('telescope','ZenithAngle'):
            zenithAngle = self.get_config_value('telescope','ZenithAngle')
        else:
            zenithAngle = 0.0
        
        airmass = 1/np.cos(zenithAngle*np.pi/180)
        
        if self.check_config_key('telescope','ObscurationRatio'):
            obsRatio = self.get_config_value('telescope','ObscurationRatio')
        else:
            obsRatio = 0.0
        
        if self.check_config_key('telescope','Resolution'):
            nPup = self.get_config_value('telescope','Resolution')
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the pupil (telescope) resolution\n')
            self.error = True
            return
            
        #----- PUPIL
        if self.check_config_key('telescope','PathPupil'):
            path_pupil = path_root + self.get_config_value('telescope','PathPupil')
        else:
            path_pupil = ''
                  
        if self.check_config_key('telescope','PupilAngle'):
            pupilAngle = self.get_config_value('telescope','PupilAngle')
        else:
            pupilAngle = 0.0
        
        if self.check_config_key('telescope','PathStaticOn'):
            path_static_on = path_root + self.get_config_value('telescope','PathStaticOn')
        else:
            path_static_on = None       
        
        if self.check_config_key('telescope','PathStaticOff'):
            path_static_off = path_root +self.get_config_value('telescope','PathStaticOff')
        else:
            path_static_off = None
        
        if self.check_config_key('telescope','PathStaticPos'):
            path_static_pos = path_root + self.get_config_value('telescope','PathStaticPos')
        else:
            path_static_pos = None
            
        #----- APODIZER
        if self.check_config_key('telescope','PathApodizer'):
            path_apodizer = path_root + self.get_config_value('telescope','PathApodizer')
        else:
            path_apodizer = ''
                
        #----- TELESCOPE ABERRATIONS
        if self.check_config_key('telescope', 'PathStatModes'):
            path_statModes = path_root + self.get_config_value('telescope','PathStatModes')
        else:
            path_statModes = ''
            
        # ----- class definition     
        self.tel = telescope(D, nPup,
                             zenith_angle=zenithAngle,
                             obsRatio=obsRatio,
                             pupilAngle=pupilAngle,
                             path_pupil=path_pupil,
                             path_static_on=path_static_on,
                             path_static_off=path_static_off,
                             path_static_pos=path_static_pos,
                             path_apodizer=path_apodizer,
                             path_statModes=path_statModes)                     

        #%% ATMOSPHERE
        
        if self.check_config_key('atmosphere','Wavelength'):
            wvlAtm = self.get_config_value('atmosphere','Wavelength') 
        else:
            wvlAtm = 500e-9
        
        if self.check_config_key('atmosphere','Seeing'):
            r0 = 0.976*wvlAtm/self.get_config_value('atmosphere','Seeing')*3600*180/np.pi 
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the atmosphere seeing\n')
            self.error = True
            return
        
        if self.check_config_key('atmosphere','L0'):
            L0 = self.get_config_value('atmosphere','L0') 
        else:
            L0 = 25
            
        if self.check_config_key('atmosphere','Cn2Weights'):
            weights = self.get_config_value('atmosphere','Cn2Weights') 
        else:
            weights = [1.0]
        
        if self.check_config_key('atmosphere','Cn2Heights'):
            heights = self.get_config_value('atmosphere','Cn2Heights') 
        else:
            heights = [0.0]
            
        if self.check_config_key('atmosphere','WindSpeed'):
            wSpeed = self.get_config_value('atmosphere','WindSpeed') 
        else:
            wSpeed = [10.0]
            
        if self.check_config_key('atmosphere','WindDirection'):
            wDir = self.get_config_value('atmosphere','WindDirection') 
        else:
            wDir = [0.0]

        #-----  verification
        if not (len(weights) == len(heights) == len(wSpeed) == len(wDir)):
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of atmospheric layers is not consistent in the parameters file\n')
            self.error = True
            return
        #----- class definition
        self.atm = atmosphere(wvlAtm, r0*airmass**(-3.0/5.0),
                              weights,
                              np.array(heights)*airmass,
                              wSpeed,
                              wDir,
                              L0)            
        
         #%%  GUIDE STARS 
        if self.check_config_key('sources_HO','Wavelength'):
            wvlGs     = np.unique(np.array(self.get_config_value('sources_HO','Wavelength')))
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the wavelength of the HO source (sources_HO)\n')
            return 0
        
        if self.check_config_key('sources_HO','Zenith'):
            zenithGs = self.get_config_value('sources_HO','Zenith') 
        else:
            zenithGs = [0.0]
            
        if self.check_config_key('sources_HO','Azimuth'):
            azimuthGs = self.get_config_value('sources_HO','Azimuth') 
        else:
            azimuthGs = [0.0]
            
        if self.check_config_key('sources_HO','Height'):
            heightGs  = self.get_config_value('sources_HO','Height') 
        else:
            heightGs = 0.0
                         
        # ----- verification
        if len(zenithGs) != len(azimuthGs):
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of guide stars for high-order sensing is not consistent in the parameters file\n')
            self.error = True
            return
        # ----- creating the source class
        if heightGs == 0:
            self.ngs = source(wvlGs,
                              zenithGs,azimuthGs,
                              tag="NGS",verbose=True)   
            self.lgs = None
        else:
            self.lgs = source(wvlGs,
                              zenithGs,azimuthGs,
                              height=heightGs*airmass,
                              tag="LGS",verbose=True)  
            
            if (not self.check_section_key('sources_LO')) | (not self.check_section_key('sources_LO')):
                print('%%%%%%%% WARNING %%%%%%%%')
                print('No information about the tip-tilt star can be retrieved\n')
                self.ngs = None
            else:
                if self.check_config_key('sources_LO','Wavelength'):
                    wvlGs = np.unique(np.array(self.get_config_value('sources_LO','Wavelength')))
                else:
                    print('%%%%%%%% ERROR %%%%%%%%')
                    print('You must provide a value for the wavelength of the LO source (sources_LO)\n')
                    self.error = True
                    return
        
                zenithGs   = np.array(self.get_config_value('sources_LO','Zenith'))
                azimuthGs  = np.array(self.get_config_value('sources_LO','Azimuth'))
                # ----- verification
                if len(zenithGs) != len(azimuthGs):
                    print('%%%%%%%% ERROR %%%%%%%%')
                    print('The number of guide stars for high-order sensing is not consistent in the parameters file\n')
                    self.error = True
                    return
                self.ngs = source(wvlGs,zenithGs,azimuthGs,tag="NGS",verbose=True)   
                              
                
        #%%  SCIENCE SOURCES
        
        if self.check_config_key('sources_science','Wavelength'):
            wvlSrc = np.array(self.get_config_value('sources_science','Wavelength'))
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the wavelength of the science source (sources_science)\n')
            self.error = True
            return
        
        if self.check_config_key('sources_science','Zenith'):
            zenithSrc = self.get_config_value('sources_science','Zenith') 
        else:
            zenithSrc = [0.0]
            
        if self.check_config_key('sources_science','Azimuth'):
            azimuthSrc = self.get_config_value('sources_science','Azimuth') 
        else:
            azimuthSrc = [0.0]
        
        if np.any(coo_stars):
            zenithSrc = np.hypot(coo_stars[0],coo_stars[1])
            azimuthSrc = np.arctan2(coo_stars[0],coo_stars[1])
        
        #----- verification
        if len(zenithSrc) != len(azimuthSrc):
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of scientific sources is not consistent in the parameters file\n')
            self.error = True
            return
        
        if self.getPSDatNGSpositions and self.check_config_key('sources_LO','Wavelength'):
            zenithSrc = zenithSrc +  (self.get_config_value('sources_LO','Zenith'))
            azimuthSrc = azimuthSrc + (self.get_config_value('sources_LO','Azimuth'))
            
            
        #----- class definition
        self.src = source(wvlSrc,
                          zenithSrc, azimuthSrc,
                          tag="SCIENCE",verbose=True)   
 
       
#%% HIGH-ORDER WAVEFRONT SENSOR
        
        if self.check_config_key('sensor_HO','PixelScale'):
            psInMas = self.get_config_value('sensor_HO','PixelScale')
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the HO detector (sensor_HO) pixel scale\n')
            self.error = True
            return
        
        if self.check_config_key('sensor_HO','FieldOfView'):
            fov = self.get_config_value('sensor_HO','FieldOfView')
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the HO detector (sensor_HO) field of view\n')
            self.error = True
            return
        
        if self.check_config_key('sensor_HO','Binning'):
            Binning = self.get_config_value('sensor_HO','Binning')
        else:
            Binning = 1
            
        if self.check_config_key('sensor_HO','SpotFWHM'):
            spotFWHM = self.get_config_value('sensor_HO','SpotFWHM')
        else:
            spotFWHM = [[0.0, 0.0]]
            
        if self.check_config_key('sensor_HO','NumberPhotons'):
            nph = self.get_config_value('sensor_HO','NumberPhotons')
        else:
            nph = [np.inf]
        
        if self.check_config_key('sensor_HO','SigmaRON'):
            ron = self.get_config_value('sensor_HO','SigmaRON')
        else:
            ron = 0.0
        
        if self.check_config_key('sensor_HO','Gain'):
            Gain = self.get_config_value('sensor_HO','Gain')
        else:
            Gain = 1
            
        if self.check_config_key('sensor_HO','SkyBackground'):
            sky = self.get_config_value('sensor_HO','SkyBackground')
        else:
            sky = 0.0
        
        if self.check_config_key('sensor_HO','Dark'):
            dark = self.get_config_value('sensor_HO','Dark')
        else:
            dark = 0.0
            
        if self.check_config_key('sensor_HO','ExcessNoiseFactor'):
            excess = self.get_config_value('sensor_HO','ExcessNoiseFactor')
        else:
            excess = 1.0
         
        if self.check_config_key('sensor_HO','SpectralBandwidth'):
            bw = self.get_config_value('sensor_HO','SpectralBandwidth')
        else:
            bw = 0.0
        if self.check_config_key('sensor_HO','Transmittance'):
            tr = self.get_config_value('sensor_HO','Transmittance')
        else:
            tr = [1.0]
        if self.check_config_key('sensor_HO','Dispersion'):
            disp = self.get_config_value('sensor_HO','Dispersion')
        else:
            disp = [[0.0], [0.0]]
                
        if self.check_config_key('sensor_HO','WfsType'):
            wfstype = self.get_config_value('sensor_HO','WfsType')
        else:
            wfstype = 'Shack-Hartmann'
            
        if self.check_config_key('sensor_HO','NumberLenslets'):
            nL = self.get_config_value('sensor_HO','NumberLenslets')
        else:
            nL = [20]
            
        if self.check_config_key('sensor_HO','SizeLenslets'):
            dsub = self.get_config_value('sensor_HO','SizeLenslets')
        else:
            dsub = list(D/np.array(nL))
            
        if self.check_config_key('sensor_HO','Modulation'):
            modu = self.get_config_value('sensor_HO','Modulation')
        else:
            modu = None
            
        if self.check_config_key('sensor_HO','NoiseVariance'):
            NoiseVar = self.get_config_value('sensor_HO','NoiseVariance')
        else:
            NoiseVar = [None]
            
        if self.check_config_key('sensor_HO','Algorithm'):
            algorithm = self.get_config_value('sensor_HO','Algorithm')
        else:
            algorithm = 'wcog'
            
        if self.check_config_key('sensor_HO','WindowRadiusWCoG'):
            wr = self.get_config_value('sensor_HO','WindowRadiusWCoG')
        else:
            wr = 5.0
            
        if self.check_config_key('sensor_HO','ThresholdWCoG = 0.0'):
            thr = self.get_config_value('sensor_HO','ThresholdWCoG = 0.0')
        else:
            thr = 0.0
            
        if self.check_config_key('sensor_HO','NewValueThrPix'):
            nv = self.get_config_value('sensor_HO','NewValueThrPix')
        else:
            nv = 0.0
         
        if self.check_config_key('sensor_HO','ExcessNoiseFactor'):
            excess = self.get_config_value('sensor_HO','ExcessNoiseFactor')
        else:
            excess = 1.0
            
        self.wfs = sensor(psInMas, fov,
                          binning=Binning, spotFWHM=spotFWHM,
                          nph=nph, bandwidth=bw, transmittance=tr, dispersion=disp,
                          gain=Gain, ron=ron, sky=sky, dark=dark, excess=excess,
                          nL=nL, dsub=dsub, wfstype=wfstype, modulation=modu,
                          noiseVar=NoiseVar, algorithm=algorithm,
                          algo_param=[wr,thr,nv], tag="HO WFS")
#%% TIP-TILT SENSORS
        if self.check_section_key('sensor_LO'):
            
            if self.check_config_key('sensor_LO','PixelScale'):
                psInMas = self.get_config_value('sensor_LO','PixelScale')
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the LO detector (sensor_LO) pixel scale\n')
                self.error = True
                return
            
            if self.check_config_key('sensor_LO','FieldOfView'):
                fov = self.get_config_value('sensor_LO','FieldOfView')
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the LO detector (sensor_LO) field of view\n')
                self.error = True
                return
            
            if self.check_config_key('sensor_LO','Binning'):
                Binning = self.get_config_value('sensor_LO','Binning')
            else:
                Binning = 1
            
            if self.check_config_key('sensor_LO','SpotFWHM'):
                spotFWHM = self.get_config_value('sensor_LO','SpotFWHM')
            else:
                spotFWHM = [[0.0, 0.0]]
                
            if self.check_config_key('sensor_LO','NumberPhotons'):
                nph = self.get_config_value('sensor_LO','NumberPhotons')
            else:
                nph = np.inf
            
            if self.check_config_key('sensor_LO','SigmaRON'):
                ron = self.get_config_value('sensor_LO','SigmaRON')
            else:
                ron = 0.0
            
            if self.check_config_key('sensor_LO','SkyBackground'):
                sky = self.get_config_value('sensor_LO','SkyBackground')
            else:
                sky = 0.0
            
            if self.check_config_key('sensor_LO','Dark'):
                dark = self.get_config_value('sensor_LO','Dark')
            else:
                dark = 0.0
                
            if self.check_config_key('sensor_LO','ExcessNoiseFactor'):
                excess = self.get_config_value('sensor_LO','ExcessNoiseFactor')
            else:
                excess = 1.0
            
            if self.check_config_key('sensor_LO','SpectralBandwidth'):
                bw = self.get_config_value('sensor_LO','SpectralBandwidth')
            else:
                bw = 0.0
            if self.check_config_key('sensor_LO','Transmittance'):
                tr = self.get_config_value('sensor_LO','Transmittance')
            else:
                tr = [1.0]
            if self.check_config_key('sensor_LO','Dispersion'):
                disp = self.get_config_value('sensor_LO','Dispersion')
            else:
                disp = [[0.0], [0.0]]
            
            if self.check_config_key('sensor_LO','NumberLenslets'):
                nL = self.get_config_value('sensor_LO','NumberLenslets')
            else:
                nL = [1]
                     
            if self.check_config_key('sensor_LO','NoiseVariance'):
                NoiseVar = self.get_config_value('sensor_LO','NoiseVariance')
            else:
                NoiseVar = [None]
                
            if self.check_config_key('sensor_LO','Algorithm'):
                algorithm = self.get_config_value('sensor_LO','Algorithm')
            else:
                algorithm = 'wcog'
                
            if self.check_config_key('sensor_LO','WindowRadiusWCoG'):
                wr = self.get_config_value('sensor_LO','WindowRadiusWCoG')
            else:
                wr = 5.0
                
            if self.check_config_key('sensor_LO','ThresholdWCoG = 0.0'):
                thr = self.get_config_value('sensor_LO','ThresholdWCoG = 0.0')
            else:
                thr = 0.0
                
            if self.check_config_key('sensor_LO','NewValueThrPix'):
                nv = self.get_config_value('sensor_LO','NewValueThrPix')
            else:
                nv = 0.0
                
                
            self.tts = sensor(psInMas, fov,
                              binning=Binning, spotFWHM=spotFWHM,
                              nph=nph, bandwidth=bw, transmittance=tr, dispersion=disp,
                              gain=Gain, ron=ron, sky=sky, dark=dark, excess=excess,
                              nL=nL, dsub=list(D/np.array(nL)),
                              wfstype='Shack-Hartmann', noiseVar=NoiseVar,
                              algorithm=algorithm, algo_param=[wr,thr,nv], tag="TT WFS")
        else:
            self.tts = None

 #%% REAL-TIME-COMPUTER
     
        if self.check_config_key('RTC','LoopGain_HO'):
            LoopGain_HO = self.get_config_value('RTC','LoopGain_HO')
        else:
            LoopGain_HO = 0.5
            
        if self.check_config_key('RTC','SensorFrameRate_HO'):
            frameRate_HO = self.get_config_value('RTC','SensorFrameRate_HO')
        else:
            frameRate_HO = 500.0
            
        if self.check_config_key('RTC','LoopDelaySteps_HO'):
            delay_HO = self.get_config_value('RTC','LoopDelaySteps_HO')
        else:
            delay_HO = 2
                     
        if self.check_config_key('RTC','LoopGain_LO'):
            LoopGain_LO = self.get_config_value('RTC','LoopGain_LO')
        else:
            LoopGain_LO = None
            
        if self.check_config_key('RTC','SensorFrameRate_LO'):
            frameRate_LO = self.get_config_value('RTC','SensorFrameRate_LO')
        else:
            frameRate_LO = None
            
        if self.check_config_key('RTC','LoopDelaySteps_LO'):
            delay_LO = self.get_config_value('RTC','LoopDelaySteps_LO')
        else:
            delay_LO = None
            
        if self.check_config_key('RTC','ResidualError'):
            wfe = self.get_config_value('RTC','ResidualError')
        else:
            wfe = None
            
        self.rtc = rtc(LoopGain_HO, frameRate_HO, delay_HO, wfe=wfe,
                       loopGainLO=LoopGain_LO, frameRateLO=frameRate_LO, delayLO=delay_LO)
               
#%% DEFORMABLE MIRRORS
        if self.check_config_key('DM','NumberActuators'):
            nActu = self.get_config_value('DM','NumberActuators')
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the Dm number of actuators (NumberActuators)\n')
            self.error = True
            return
        
        if self.check_config_key('DM','DmPitchs'):
            DmPitchs = np.array(self.get_config_value('DM','DmPitchs'))
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the Dm actuators pitch (DmPitchs)\n')
            self.error = True
            return
        
        if self.check_config_key('DM','InfModel'):
            InfModel = self.get_config_value('DM','InfModel') 
        else:
            InfModel = 'gaussian'
            
        if self.check_config_key('DM','InfCoupling'):
            InfCoupling = self.get_config_value('DM','InfCoupling') 
        else:
            InfCoupling = [0.2]
            
        if self.check_config_key('DM','DmHeights'):
            DmHeights = self.get_config_value('DM','DmHeights') 
        else:
            DmHeights = [0.0]
            
        if self.check_config_key('DM','OptimizationWeight'):
            opt_w = self.get_config_value('DM','OptimizationWeight') 
        else:
            opt_w = [0.0]
            
        if self.check_config_key('DM','OptimizationAzimuth'):
            opt_az = self.get_config_value('DM','OptimizationAzimuth') 
        else:
            opt_az = [0.0]
            
        if self.check_config_key('DM','OptimizationZenith'):
            opt_zen = self.get_config_value('DM','OptimizationZenith') 
        else:
            opt_zen = [0.0]

         # ----- verification
        if (len(opt_zen) != len(opt_az)) or (len(opt_zen) != len(opt_w)):
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of optimization directions is not consistent in the parameters file\n')
            self.error = True
            return
              
        if self.check_config_key('DM','OptimizationConditioning'):
            cond = self.get_config_value('DM','OptimizationConditioning') 
        else:
            cond = 100.0
            
        if self.check_config_key('DM','NumberReconstructedLayers'):
            nrec = self.get_config_value('DM','NumberReconstructedLayers') 
        else:
            nrec = 10
            
        if self.check_config_key('DM','AoArea'):
            AoArea = self.get_config_value('DM','AoArea') 
        else:
            AoArea = 'circle'
        
        # ----- creating the dm class
        self.dms = deformableMirror(nActu, DmPitchs,
                                    heights=DmHeights, mechCoupling=InfCoupling,
                                    modes=InfModel,
                                    opt_dir=[opt_zen,opt_az],
                                    opt_weights=opt_w,
                                    opt_cond=cond,n_rec = nrec,
                                    AoArea=AoArea)
      
#%% SCIENCE DETECTOR
        
        if self.check_config_key('sensor_science','Name'):
            camName = self.get_config_value('sensor_science','Name')
        else:
            camName = 'SCIENCE CAM'
            
        if self.check_config_key('sensor_science','PixelScale'):
            psInMas = self.get_config_value('sensor_science','PixelScale')
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the science detector (sensor_science) pixel scale\n')
            self.error = True
            return
        
        if self.check_config_key('sensor_science','FieldOfView'):
            fov = self.get_config_value('sensor_science','FieldOfView')
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the science detector (sensor_science) field of view\n')
            self.error = True
            return
        
        if self.check_config_key('sensor_science','Binning'):
            Binning = self.get_config_value('sensor_science','Binning')
        else:
            Binning = 1
            
        if self.check_config_key('sensor_science','SpotFWHM'):
            spotFWHM = self.get_config_value('sensor_science','SpotFWHM')
        else:
            spotFWHM = [[0.0,0.0,0.0]]
            
        if self.check_config_key('sensor_science','SpectralBandwidth'):
            bw = self.get_config_value('sensor_science','SpectralBandwidth')
        else:
            bw = 0.0
        if self.check_config_key('sensor_science','Transmittance'):
            tr = self.get_config_value('sensor_science','Transmittance')
        else:
            tr = [1.0]
        if self.check_config_key('sensor_science','Dispersion'):
            disp = self.get_config_value('sensor_science','Dispersion')
        else:
            disp = [[0.0],[0.0]]
        
        if self.check_config_key('sensor_science','NumberPhotons'):
            nph = self.get_config_value('sensor_science','NumberPhotons')
        else:
            nph = np.inf
        
        if self.check_config_key('sensor_science','Saturation'):
            saturation = self.get_config_value('sensor_science','Saturation')
        else:
            saturation = np.inf
            
        if self.check_config_key('sensor_science','SigmaRON'):
            ron = self.get_config_value('sensor_science','SigmaRON')
        else:
            ron = 0.0
        
        if self.check_config_key('sensor_science','Gain'):
            Gain = self.get_config_value('sensor_science','Gain')
        else:
            Gain = 1
            
        if self.check_config_key('sensor_science','SkyBackground'):
            sky = self.get_config_value('sensor_science','SkyBackground')
        else:
            sky = 0.0
        
        if self.check_config_key('sensor_science','Dark'):
            dark = self.get_config_value('sensor_science','Dark')
        else:
            dark = 0.0
        
        if self.check_config_key('sensor_science','ExcessNoiseFactor'):
            excess = self.get_config_value('sensor_science','ExcessNoiseFactor')
        else:
            excess = 1.0
        
        self.cam = detector(psInMas, fov,
                            binning=Binning, spotFWHM=spotFWHM, saturation=saturation,
                            nph=nph, bandwidth=bw, transmittance=tr, dispersion=disp,
                            gain=Gain, ron=ron, sky=sky, dark=dark, excess=excess,
                            tag=camName)
        
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
        
