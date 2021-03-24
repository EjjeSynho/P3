#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:50:20 2021

@author: omartin
"""

#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
import numpy.fft as fft
import time
import sys as sys
import os.path as ospath
from configparser import ConfigParser
import scipy.special as ssp
import re
from scipy.ndimage import rotate

from astropy.io import fits

import fourier.FourierUtils as FourierUtils
from aoSystem.telescope import telescope
from aoSystem.atmosphere import atmosphere
from aoSystem.source import source

#%%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000

class psfao21:
    
    # WAVELENGTH
    @property
    def wvl(self):
        return self.__wvl
    @wvl.setter
    def wvl(self,val):
        self.__wvl = val
        self.samp  = val* rad2mas/(self.psInMas*self.D)
    @property
    def wvlRef(self):
        return self.__wvlRef
    @wvlRef.setter
    def wvlRef(self,val):
        self.__wvlRef = val
        self.sampRef  = val* rad2mas/(self.psInMas*self.D)
    
    @property
    def wvlCen(self):
        return self.__wvlCen
    @wvlCen.setter
    def wvlCen(self,val):
        self.__wvlCen = val
        self.sampCen  = val* rad2mas/(self.psInMas*self.D)
        
    # SAMPLING
    @property
    def samp(self):
        return self.__samp
    @samp.setter
    def samp(self,val):
        self.k_      = np.ceil(2.0/val).astype('int') # works for oversampling
        self.__samp  = self.k_ * val     
    @property
    def sampCen(self):
        return self.__sampCen
    @sampCen.setter
    def sampCen(self,val):
        self.kCen_      = int(np.ceil(2.0/val))# works for oversampling
        self.__sampCen  = self.kCen_ * val  
    @property
    def sampRef(self):
        return self.__sampRef
    @sampRef.setter
    def sampRef(self,val):
        self.kRef_      = int(np.ceil(2.0/val)) # works for oversampling
        self.__sampRef  = self.kRef_ * val
        self.kxky_      = self.freq_array(self.nPix*self.kRef_,self.__sampRef,self.D)
        self.k2_        = self.kxky_[0]**2 + self.kxky_[1]**2                   
        #piston filter
        if hasattr(self,'Dcircle'):
            D  = self.Dcircle
        else:
            D = self.D          
        self.pistonFilter_= 1 - 4*self.sombrero(1, np.pi*D*np.sqrt(self.k2_)) ** 2
        self.pistonFilter_[self.nPix*self.kRef_//2,self.nPix*self.kRef_//2] = 0
            
    # CUT-OFF FREQUENCY
    @property
    def nAct(self):
        return self.__nAct    
    @nAct.setter
    def nAct(self,val):
        self.__nAct = val
        # redefining the ao-corrected area
        self.kc_= (val-1)/(2.0*self.D)
        kc2     = self.kc_**2
        if self.circularAOarea:
            self.mskOut_   = (self.k2_ >= kc2)
            self.mskIn_    = (self.k2_ < kc2)
        else:
            self.mskOut_   = np.logical_or(abs(self.kxky_[0]) >= self.kc_, abs(self.kxky_[1]) >= self.kc_)
            self.mskIn_    = np.logical_and(abs(self.kxky_[0]) < self.kc_, abs(self.kxky_[1]) < self.kc_)
        self.psdKolmo_     = 0.0229 * self.mskOut_* ((1.0 /self.atm.L0**2) + self.k2_) ** (-11.0/6.0)
        self.wfe_fit_norm  = np.sqrt(np.trapz(np.trapz(self.psdKolmo_,self.kxky_[1][0]),self.kxky_[1][0]))
    # INIT
    def __init__(self,file,circularAOarea='circle',antiAlias=False,aliasPSD=False,pathStat=None,fitCn2=False):
        
        tstart = time.time()
        
        # PARSING INPUTS
        self.file      = file
        self.antiAlias = antiAlias
        self.pathStat  = pathStat
        self.status = self.parameters(self.file,circularAOarea=circularAOarea)
        
        if self.status:
            # DEFINING THE DOMAIN ANGULAR FREQUENCIES
            self.nOtf        = self.nPix * self.kRef_
            self.U_,self.V_  = self.shift_array(self.nOtf,self.nOtf,fact = 2)     
            self.U2_         = self.U_**2
            self.V2_         = self.V_**2
            self.UV_         = self.U_*self.V_
            self.otfDL       = self.getStaticOTF(self.nOtf,self.sampRef,self.tel.obsRatio,self.wvlRef)
            self.otfStat     = self.otfDL
            
            # ALIASING PSD
            self.aliasPSD = aliasPSD
            if self.aliasPSD:
                self.psdAlias = self.aliasingPSD()
            else:
                self.psdAlias = 0
                
            # ANISOPLANATISM
            if any(self.atm.heights) and any(self.src.zenith):
                self.Dani_l = self.instantiateAnisoplanatism(self.nOtf,self.sampRef)
                self.isAniso = True
                if fitCn2 == False: 
                    self.Kaniso = np.zeros((self.src.nSrc,self.nWvl,self.nOtf,self.nOtf))
                    Cn2  = self.atm.r0**(-5/3) * (self.atm.wvl/self.wvlRef)**2  * self.atm.weights
                    for l in range(self.nWvl):
                        wvl_l     = self.wvl[l]
                        wvlRatio  = (self.wvlRef/wvl_l) ** 2
                        for iSrc in range(self.src.nSrc):   
                            Dani                = (self.Dani_l[iSrc].T * Cn2 * wvlRatio).sum(axis=2)
                            self.Kaniso[iSrc,l] = np.exp(-0.5 * Dani)
                    self.atm = atmosphere(self.atm.wvl,self.atm.r0,[1],[0],self.atm.wSpeed.mean(),self.atm.wDir.mean(),self.atm.L0)   
                    self.isAniso = False
            else:
                self.Dani_l = None
                self.Kaniso = np.ones((self.src.nSrc,self.nWvl))
                self.isAniso = False
                if fitCn2 == False: 
                    self.atm = atmosphere(self.atm.wvl,self.atm.r0,[1],[0],self.atm.wSpeed.mean(),self.atm.wDir.mean(),self.atm.L0)  
            
            # ONE OF SEVERAL FRAMES
            self.isCube = any(rad2mas * self.src.direction[0]/self.psInMas > self.nPix) \
            or all(rad2mas * self.src.direction[1]/self.psInMas > self.nPix)
                
            # DEFINING BOUNDS
            self.bounds = self.defineBounds()
        self.t_init = 1000*(time.time()  - tstart)
        
    def _repr__(self):
        return 'PSFAO21 model'
    
    
    def parameters(self,file,circularAOarea='circle',Dcircle=None):
                    
        tstart = time.time() 
    
        # verify if the file exists
        if ospath.isfile(file) == False:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The .ini file does not exist\n')
            return 0
        
        # open the .ini file
        config = ConfigParser()
        config.optionxform = str
        config.read(file)
        
        #%% Telescope
        self.D         = eval(config['telescope']['TelescopeDiameter'])
        if config.has_option('telescope','obscurationRatio'):
            obsRatio = eval(config['telescope']['obscurationRatio'])
        else:
            obsRatio = 0.0
        if config.has_option('telescope','zenithAngle'):
            zenithAngle = eval(config['telescope']['zenithAngle'])
        else:
            zenithAngle = 0.0
        if config.has_option('telescope','path_pupil'):
            path_pupil     = eval(config['telescope']['path_pupil'])
        else:
            path_pupil = []
        if config.has_option('telescope','path_static'):
            path_static = eval(config['telescope']['path_static'])
        else:
            path_static = []         
        if config.has_option('telescope','path_apodizer'):
            path_apodizer  = eval(config['telescope']['path_apodizer'])
        else:
            path_apodizer = []           
        if config.has_option('telescope','CircleDiameter'):
            self.Dcircle = eval(config['telescope']['CircleDiameter'])
        else:
            self.Dcircle = self.D          
        if config.has_option('telescope','pupilAngle'):
            self.pupilAngle= eval(config['telescope']['pupilAngle'])
        else:
            self.pupilAngle= 0.0
    
        #%% Atmosphere
        wvlAtm         = eval(config['atmosphere']['atmosphereWavelength']) 
        r0             = 0.976*wvlAtm/eval(config['atmosphere']['seeing'])*3600*180/np.pi 
        L0             = eval(config['atmosphere']['L0']) 
        weights        = np.array(eval(config['atmosphere']['Cn2Weights']) )
        heights        = np.array(eval(config['atmosphere']['Cn2Heights']) )
        wSpeed         = np.array(eval(config['atmosphere']['wSpeed']) )
        wDir           = np.array(eval(config['atmosphere']['wDir']) )
        #-----  verification
        if not (len(weights) == len(heights) == len(wSpeed) == len(wDir)):
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of atmospheric layers is not consistent in the parameters file\n')
            return 0
        self.atm = atmosphere(wvlAtm,r0,weights,heights/np.cos(zenithAngle*np.pi/180),wSpeed,wDir,L0)            
        
        #%% Sampling and field of view
        self.psInMas = eval(config['PSF_DIRECTIONS']['psInMas'])
        self.circularAOarea= circularAOarea
        self.nPix    = eval(config['PSF_DIRECTIONS']['psf_FoV'])
        
        #%% PSF directions
        wvlSrc         = np.array(eval(config['PSF_DIRECTIONS']['ScienceWavelength']))
        zenithSrc      = np.array(np.array(eval(config['PSF_DIRECTIONS']['ScienceZenith'])))
        azimuthSrc     = np.array(np.array(eval(config['PSF_DIRECTIONS']['ScienceAzimuth'])))        
        # ----- verification
        self.src = []
        if len(zenithSrc) != len(azimuthSrc):
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of scientific sources is not consistent in the parameters file\n')
            return 0
        
        # MANAGING WAVELENGTHS
        if config.has_option('POLYCHROMATISM','spectralBandwidth'):
            self.wvl_bw = eval(config['POLYCHROMATISM']['spectralBandwidth'])
        else:
            self.wvl_bw = [0]
        
        if self.wvl_bw != [0]:
            # CASE 1: SINGLE POLYCHROMATIC PSF
            self.wvl_dpx = np.array(eval(config['POLYCHROMATISM']['dispersionX']))
            self.wvl_dpy = np.array(eval(config['POLYCHROMATISM']['dispersionY']))
            self.wvl_tr  = np.array(eval(config['POLYCHROMATISM']['transmittance']))
            if (len(self.wvl_dpx) == len(self.wvl_dpx)) and (len(self.wvl_dpx) == len(self.wvl_tr)):
                self.nWvl    = len(self.wvl_tr)
                self.wvlRef  = wvlSrc - self.wvl_bw/2
                self.wvlMax  = wvlSrc + self.wvl_bw/2
                self.wvl     = np.linspace(self.wvlRef,self.wvlMax,self.nWvl)
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('The number of spectral bins are not consistent in the parameters file\n')
                return 0
        else:
            # CASE 2 : DATA CUBE OF MONOCHROMATIC PSF
            self.wvl     = np.unique(wvlSrc)
            self.nWvl    = len(self.wvl)
            self.wvl_tr  = np.ones(self.nWvl)
            self.wvl_dpx = np.zeros(self.nWvl)
            self.wvl_dpy = np.zeros(self.nWvl)
            self.wvlRef  = self.wvl.min()
            
            
        self.wvlCen = np.mean(self.wvl)
        #PUPIL RESOLUTION
        if config.has_option('telescope','resolution'):
            self.nPup = eval(config['telescope']['resolution'])
        else:
            self.nPup = 2*int(np.ceil(self.nPix*self.kRef_/self.samp.min()/2))
                
        self.nAct    = np.floor(self.D/np.array(eval(config['DM']['DmPitchs']))+1)
        self.src     = source(wvlSrc,zenithSrc,azimuthSrc,types="SCIENTIFIC STAR",verbose=True)   
        
        #%% UPDATING PUPIL RESOLUTION
        self.tel     = telescope(self.D,zenithAngle,obsRatio,self.nPup,pupilAngle = self.pupilAngle,file=path_pupil)                     
    
        # APODIZER
        self.apodizer = 1
        if path_apodizer != [] and ospath.isfile(path_apodizer) == True:
             if  re.search(".fits",path_apodizer)!=None :
                self.apodizer = fits.getdata(path_apodizer)
                self.apodizer = FourierUtils.interpolateSupport(self.apodizer,self.nPup,kind='linear')
                
        # EXTERNAL STATIC MAP
        self.opdMap_ext = 0
        if path_static != [] and ospath.isfile(path_static) == True:
             if  re.search(".fits",path_static)!=None :
                self.opdMap_ext = fits.getdata(path_static)
                self.opdMap_ext = self.tel.pupil*FourierUtils.interpolateSupport(self.opdMap_ext,self.nPup,kind='linear')
        
        # STATIC ABERRATIONS
        self.statModes = None
        self.nModes = 1
        self.isStatic = False
        if config.has_option('telescope', 'path_statModes'):
            self.path_statModes = eval(config['telescope']['path_statModes'])
        else:
            self.path_statModes = []
            
        if self.path_statModes:
            if ospath.isfile(self.path_statModes) == True and re.search(".fits",self.path_statModes)!=None:                
                self.statModes = fits.getdata(self.path_statModes)
                s1,s2,s3 = self.statModes.shape
                if s1 != s2: # mode on first dimension
                    tmp = np.transpose(self.statModes,(1,2,0))  
                else:
                    tmp = self.statModes
                    
                self.nModes = tmp.shape[-1]
                self.statModes = np.zeros((self.nPup,self.nPup,self.nModes))
                
                for k in range(self.nModes):
                    mode = FourierUtils.interpolateSupport(tmp[:,:,k],self.nPup,kind='linear')
                    if self.pupilAngle !=0:
                        mode = rotate(mode,self.pupilAngle,reshape=False)
                    self.statModes[:,:,k] = self.tel.pupil * mode
                self.isStatic = True
        
        #%% Guide stars
        wvlGs      = np.unique(np.array(eval(config['SENSOR_HO']['SensingWavelength_HO'])))
        zenithGs   = np.array(eval(config['GUIDESTARS_HO']['GuideStarZenith_HO']))
        azimuthGs  = np.array(eval(config['GUIDESTARS_HO']['GuideStarAzimuth_HO']))
        heightGs   = eval(config['GUIDESTARS_HO']['GuideStarHeight_HO'])
        # ----- verification
        if len(zenithGs) != len(azimuthGs):
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of guide stars for high-order sensing is not consistent in the parameters file\n')
            return 0
        self.gs = source(wvlGs,zenithGs,azimuthGs,height=heightGs,types="GUIDE STAR",verbose=True)   

        self.t_getParam = 1000*(time.time() - tstart)
        return 1
    
    def defineBounds(self):
          #Cn2 , C , A , ax , p , theta , beta , sx , sy , sxy , F , dx , dy , bg , stat
          _EPSILON = np.sqrt(sys.float_info.epsilon)
          
          # Bounds on Cn2
          bounds_down = list(np.zeros(self.atm.nL))
          bounds_up   = list(np.inf * np.ones(self.atm.nL))          
          # PSD Parameters
          bounds_down += [0,0,_EPSILON,_EPSILON,-np.pi,1+_EPSILON]
          bounds_up   += [np.inf,np.inf,np.inf,np.inf,np.pi,5]         
          # Jitter
          bounds_down += [0,0,-1]
          bounds_up   += [np.inf,np.inf,1]
          # Photometry
          bounds_down += list(np.zeros(self.src.nSrc))
          bounds_up   += list(np.inf*np.ones(self.src.nSrc))
          # Astrometry
          bounds_down += list(-self.nPix//2 * np.ones(2*self.src.nSrc))
          bounds_up   += list( self.nPix//2 * np.ones(2*self.src.nSrc))
          # Background
          bounds_down += [-np.inf]
          bounds_up   += [np.inf]
          # Static aberrations
          bounds_down += list(-self.wvlRef/2*1e9 * np.ones(self.nModes))
          bounds_up   += list(self.wvlRef/2 *1e9 * np.ones(self.nModes))
          return (bounds_down,bounds_up)
        
        
    def getPSD(self,x0):
        # Get the moffat PSD
        pix2freq = 1/(self.tel.D * self.sampRef)
        psd = self.moffat(self.kxky_,list(x0[3:])+[0,0])
        # Piston filtering
        psd = self.pistonFilter_ * psd
        # Combination
        psd = x0[0] * (self.psdAlias+self.psdKolmo_) + self.mskIn_ * (x0[1] + psd/psd.sum() * x0[2]/pix2freq**2 )

        # Wavefront error
        self.wfe = np.sqrt( np.trapz(np.trapz(psd,self.kxky_[1][0]),self.kxky_[1][0]) ) * self.wvlRef*1e9/2/np.pi
        self.wfe_fit = np.sqrt(x0[0]) * self.wfe_fit_norm  * self.wvlRef*1e9/2/np.pi
        return psd
    
    def getStaticOTF(self,nOtf,samp,cobs,wvl,xStat=[]):
        
        # DEFINING THE RESOLUTION/PUPIL
        nPup = self.tel.pupil.shape[0]
        
        # ADDING STATIC MAP
        phaseStat = np.zeros((nPup,nPup))
        if np.any(self.opdMap_ext):
            phaseStat = (2*np.pi*1e-9/wvl) * self.opdMap_ext
            
        # ADDING USER-SPECIFIED STATIC MODES
        xStat = np.asarray(xStat)
        self.phaseMap = 0
        if self.isStatic:
            if self.statModes.shape[2]==len(xStat):
                self.phaseMap = 2*np.pi*1e-9/wvl * np.sum(self.statModes*xStat,axis=2)
                phaseStat += self.phaseMap
                
        # OTF
        otfStat = FourierUtils.pupil2otf(self.tel.pupil * self.apodizer,self.tel.pupil*phaseStat,samp)
        otfStat = FourierUtils.interpolateSupport(otfStat,nOtf)
        otfStat/= otfStat.max()
        return otfStat
    
    def __call__(self,x0,nPix=None):
        
        # INSTANTIATING
        if nPix == None:
            nPix = self.nPix
        psf = np.zeros((self.src.nSrc,nPix,nPix))
        
        # GETTING THE PARAMETERS
        xall = x0       
        # Cn2 profile
        nL   = self.atm.nL
        Cn2  = np.asarray(x0[0:nL])
        r053 = np.sum(Cn2)
        # PSD
        x0_psd = list(xall[nL:nL+6])
        # Jitter
        x0_jitter = list(xall[nL+6:nL+9])
        # Astrometry/Photometry/Background
        x0_stellar = list(xall[nL+9:nL+10+3*self.src.nSrc])
        # Static aberrations
        if self.isStatic:
            x0_stat = list(xall[nL+10+3*self.src.nSrc:])
        else:
            x0_stat = []    
        
        # GETTING THE PHASE STRUCTURE FUNCTION
        self.psd = self.getPSD([r053]+ x0_psd)
        if self.antiAlias:
            	self.psd = np.pad(self.psd,(self.nOtf//2,self.nOtf//2)) 
        #covariance map
        Bphi        = fft.fft2(fft.fftshift(self.psd)) / (self.tel.D * self.sampRef)**2       
        # phase structure function
        self.Dphi   = fft.fftshift(np.real(2*(Bphi.max() - Bphi)))
        if self.antiAlias:
            	self.Dphi   = FourierUtils.interpolateSupport(self.Dphi,self.nOtf)    

        # ADDITIONAL JITTER
        if len(x0_jitter) and (x0_jitter[0]!=0 or x0_jitter[1]!=0): # note sure about the normalization
            self.Djitter = (x0_jitter[0]**2 * self.U2_ + x0_jitter[1]**2 * self.V2_+ \
                    2*x0_jitter[0] * x0_jitter[1] * x0_jitter[2] * self.UV_)
            Kjitter = np.exp(-0.5 * self.Djitter * (np.sqrt(2)/self.psInMas)**2)
        else:
            	Kjitter = 1
                               
        for l in range(self.nWvl): # LOOP ON WAVELENGTH
            
            # WAVELENGTH
            wvl_l       = self.wvl[l]
            wvlRatio    = (self.wvlRef/wvl_l) ** 2           
            
            # STATIC OTF
            if len(x0_stat) or self.nWvl > 1:
                self.otfStat = self.getStaticOTF(int(self.nPix*self.k_[l]),self.samp[l],self.tel.obsRatio,wvl_l,xStat=x0_stat)
                self.otfStat/= self.otfStat.max()
            else:
                self.otfStat = self.otfDL
              
            # ON-AXIS OTF
            otfOn = self.otfStat * np.exp(-0.5*self.Dphi * wvlRatio)
            
            for iSrc in range(self.src.nSrc): # LOOP ON SOURCES
                
                # ANISOPLANATISM
                if self.isAniso and len(Cn2) == self.Dani_l.shape[1]:
                    Dani = (self.Dani_l[iSrc].T * Cn2 * wvlRatio).sum(axis=2)
                    Kaniso = np.exp(-0.5 * Dani)
                else:
                    Kaniso = self.Kaniso[iSrc,l]
                    
                # Stellar parameters
                if len(x0_stellar):
                    F   = x0_stellar[iSrc] * self.wvl_tr[l]
                    dx  = x0_stellar[iSrc + self.src.nSrc] + self.wvl_dpx[l]
                    dy  = x0_stellar[iSrc + 2*self.src.nSrc] + self.wvl_dpy[l]
                    bkg = x0_stellar[3*self.src.nSrc]
                else:
                    F   = self.wvl_tr[l]
                    dx  = self.wvl_dpx[l]
                    dy  = self.wvl_dpy[l]
                    bkg = 0
                
                # Phasor
                if dx !=0 or dy!=0:
                    fftPhasor = np.exp(np.pi*complex(0,1)*(self.U_*dx + self.V_*dy))
                else:
                    fftPhasor = 1
                    
                # Get the total OTF
                self.otfTot = otfOn * fftPhasor * Kaniso * Kjitter
                
                # Get the PSF
                psf_i = np.real(fft.fftshift(fft.ifft2(fft.fftshift(self.otfTot))))
                
                # CROPPING
                if self.k_[l]> 1:
                        psf_i = FourierUtils.interpolateSuport(psf_i,nPix);            
                    
                if nPix != self.nPix:
                    psf_i = FourierUtils.cropSupport(psf_i,nPix/self.nPix)
                
                # SCALING
                psf[iSrc] += F * psf_i/psf_i.sum()
        
        # DEFINING THE OUTPUT FORMAT
        if self.isCube:
            return np.squeeze(psf) + bkg
        else:
            return np.squeeze(psf.sum(axis=0)) + bkg
        
    def moffat(self,XY,x0):
        
        # parsing inputs
        a       = x0[0]
        p       = x0[1]
        theta   = x0[2]
        beta    = x0[3]
        dx      = x0[4]
        dy      = x0[5]
        
        # updating the geometry
        ax      = a*p
        ay      = a/p
        c       = np.cos(theta)
        s       = np.sin(theta)
        s2      = np.sin(2.0 * theta)
        
     
        Rxx = (c/ax)**2 + (s/ay)**2
        Ryy = (c/ay)**2 + (s/ax)**2
        Rxy =  s2/ay**2 -  s2/ax**2
            
        u = Rxx * (XY[0] - dx)**2 + Rxy * (XY[0] - dx)* (XY[1] - dy) + Ryy * (XY[1] - dy)**2
        return (1.0 + u) ** (-beta)
        
    def freq_array(self,nX,samp,D):
        k2D = np.mgrid[0:nX, 0:nX].astype(float)
        k2D[0] -= nX//2
        k2D[1] -= nX//2
        k2D *= 1.0/(D*samp)
        return k2D

    def shift_array(self,nX,nY,fact=2*np.pi*complex(0,1)):    
        X, Y = np.mgrid[0:nX,0:nY].astype(float)
        X = (X-nX/2) * fact/nX
        Y = (Y-nY/2) * fact/nY
        return X,Y
    
    def sombrero(self,n,x):
        x = np.asarray(x)
        if n==0:
            return ssp.jv(0,x)/x
        else:
            if n>1:
                out = np.zeros(x.shape)
            else:
                out = 0.5*np.ones(x.shape)
                
            out = np.zeros_like(x)
            idx = x!=0
            out[idx] = ssp.j1(x[idx])/x[idx]
            return out
        
    def instantiateAnisoplanatism(self,nPix,samp,method='psd'):
        
        def mcDonald(x):
            out = 3/5 * np.ones_like(x)
            idx  = x!=0
            if np.any(idx==False):
                out[idx] = x[idx] ** (5/6) * ssp.kv(5/6,x[idx])/(2**(5/6) * ssp.gamma(11/6))
            else:
                out = x ** (5/6) * ssp.kv(5/6,x)/(2**(5/6) * ssp.gamma(11/6))
            return out
        
        def Ialpha(x,y):
            return mcDonald(np.hypot(x,y))
        
        nSrc    = self.src.nSrc
        nLayer  = self.atm.nL
        Hs      = self.atm.heights * self.tel.airmass
        if method == 'psd':   
            # PSD METHOD : FASTER PSD METHOD WORKING FOR ANGULAR ANISOPLANATISM        
            Dani_l = np.zeros((nSrc,nLayer,nPix,nPix))
            L  = (self.tel.D * self.sampRef)**2
            
            cte  = (24*ssp.gamma(6/5)/5)**(5/6)*(ssp.gamma(11/6)**2./(2.*np.pi**(11/3)))
            kern = cte * ((1.0 /self.atm.L0**2) + self.k2_) ** (-11/6)
            kern = self.pistonFilter_ * kern
            kern[self.nOtf//2,self.nOtf//2] = 0
            for s in range(nSrc):
                ax = self.src.direction[0] - self.gs.direction[0]
                ay = self.src.direction[1] - self.gs.direction[1]
                for iSrc in range(nSrc):
                    thx = ax[iSrc]
                    thy = ay[iSrc]            
                    if thx !=0 or thy !=0:
                        for l  in range(nLayer):
                            zl = Hs[l]
                            if zl !=0:
                                psd = 2*self.mskIn_*kern*(1 - np.cos(2*np.pi*zl*(self.kxky_[0]*thx + self.kxky_[1]*thy)))    
                                cov = fft.fftshift(fft.fft2(fft.fftshift(psd)))/L
                                Dani_l[iSrc,l] = np.real(2*(cov.max()-cov))
        else:        
            # FLICKER'S MODEL - need to be generalize to focal anisoplanatism + anisokitenism
            #1\ Defining the spatial filters
            D       = self.tel.D
            f0      = 2*np.pi/self.atm.L0

            #2\ SF Calculation
            Dani_l = np.zeros((nSrc,nLayer,nPix,nPix))

            # Angular frequencies
            U,V = self.shift_array(nPix,nPix,fact=D*self.sampRef)
       
            # Instantiation
            I0      = 3/5
            I1      = Ialpha(f0*U,f0*V)
            cte     = 0.12184*0.06*(2*np.pi)**2*self.atm.L0**(5/3)

            # Anisoplanatism Structure Function
            ax = self.src.direction[0] - self.gs.direction[0]
            ay = self.src.direction[1] - self.gs.direction[1]
            for iSrc in range(nSrc):
                thx = ax[iSrc]
                thy = ay[iSrc]            
                if thx !=0 or thy !=0:
                    for l  in range(nLayer):
                        zl    = Hs[l]
                        if zl !=0: 
                            I2    = Ialpha(f0*zl*thx,f0*zl*thy)
                            I3    = Ialpha(f0 * (U + zl*thx) , f0 * (V + zl*thy))
                            I4    = Ialpha(f0 * (U - zl*thx) , f0 * (V - zl*thy))
                            Dani_l[iSrc,l]  = cte*(2*I0 - 2*I1 - 2*I2  + I3  + I4)
            
        return Dani_l
    
    def aliasingPSD(self,nTimes=2):
        """
        """
        psd = np.zeros((self.nPix*self.k_,self.nPix*self.k_))
        d   = self.tel.D/(self.nAct-1)
        i   = complex(0,1)
        Sx  = 2*i*np.pi*self.kxky_[0]*d
        Sy  = 2*i*np.pi*self.kxky_[1]*d                        
        Av  = np.sinc(d*self.kxky_[0])*np.sinc(d*self.kxky_[1])*np.exp(i*np.pi*d*(self.kxky_[0]+self.kxky_[1]))  
        SxAv  = Sx*Av
        SyAv  = Sy*Av
        gPSD  = abs(SxAv)**2 + abs(SyAv)**2 + 1e-10
        Rx    = np.conj(SxAv)/gPSD
        Ry    = np.conj(SyAv)/gPSD
        
        for mi in range(-nTimes,nTimes):
            for ni in range(-nTimes,nTimes):
                if (mi!=0) | (ni!=0):
                    km   = self.kxky_[0] - mi/d
                    kn   = self.kxky_[1] - ni/d
                    PR   = FourierUtils.pistonFilter(self.tel.D,np.hypot(km,kn),fm=mi/d,fn=ni/d)
                    W_mn = (km**2 + kn**2 + 1/self.atm.L0**2)**(-11/6)     
                    Q    = (Rx*km + Ry*kn) * (np.sinc(d*km)*np.sinc(d*kn))
                    psd = psd + PR*W_mn *abs(Q)**2
        
        psd[self.nPix*self.k_//2,self.nPix*self.k_//2] = 0
        return abs(self.pistonFilter_*self.mskIn_ * psd* 0.0229) 
