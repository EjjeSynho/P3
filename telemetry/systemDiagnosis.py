#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:04:21 2021

@author: omartin
"""

#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
import numpy.fft as fft
import aoSystem.FourierUtils as FourierUtils
from aoSystem.zernike import zernike
from astropy.io import fits
from scipy.ndimage import rotate
from scipy.optimize import least_squares
from psfFitting.confidenceInterval import confidence_interval

# 
# TO DO : 
# - verifying the Zernike reconstruction
# - WFS pupil mask
# - noise estimation, test the rtf approach
# - implement estimation of the wind speed

#%%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000
arc2rad = 1/rad2arc

class systemDiagnosis:

    def __init__(self,trs,noiseMethod='autocorrelation',nshift=1,nfit=2,\
                 noise=0,quantile=0.95,nWin=10,nZer=None,j0=4):
        
        self.trs = trs
        
        # Transfer function
        self.TemporalTransferFunction()
        
        # noise
        self.noiseMethod = noiseMethod
        if hasattr(self.trs.wfs,'intensity'):
            self.trs.wfs.nph,self.trs.wfs.ron,self.trs.tipTilt.nph,self.trs.tipTilt.ron = self.GetNumberPhotons()
        self.trs.wfs.Cn_ao,self.trs.tipTilt.Cn_tt = self.GetNoiseVariance(noiseMethod=self.noiseMethod,nshift=nshift,nfit=nfit)
        
        # GET THE VARIANCE
        Cnn = np.diag(self.trs.wfs.Cn_ao)
        Cnn = Cnn[Cnn!=0]
        self.trs.wfs.noiseVar = [(2*np.pi/self.trs.wfs.wvl)**2 *np.mean(Cnn),]*len(self.trs.wfs.nSubap)
        
        # Zernike
        self.trs.wfs.Cz_ao, self.trs.tipTilt.Cz_tt = self.ReconstructZernike(nZer=nZer,j0=j0)
        
        # Atmosphere statistics
        r0, L0, tau0, v0, dr0, dL0, dtau0, dv0 = self.GetAtmosphereStatistics(noise=noise,quantile=quantile,nWin=nWin)
        self.trs.atm.r0_tel   = r0
        self.trs.atm.L0_tel   = L0
        self.trs.atm.tau0_tel = tau0
        self.trs.atm.v0_tel   = v0
        self.trs.atm.dr0_tel  = dr0
        self.trs.atm.dL0_tel  = dL0
        self.trs.atm.dtau0_tel= dtau0
        self.trs.atm.dv0_tel  = dv0
        
        # update the atmosphere structure
        self.trs.atm.r0 = r0
        self.trs.atm.L0 = L0
        self.trs.atm.seeing = rad2arc * self.trs.atm.wvl/r0
        if len(self.trs.atm.Cn2Weights) > 1:
            vm = FourierUtils.eqLayers(self.trs.atm.Cn2Weights,self.trs.atm.wSpeed,nEqLayers=1)[1] 
            self.trs.atm.wSpeed = v0 * self.trs.atm.wSpeed/vm
        else:
            self.trs.atm.wSpeed = [v0]
        
#%%    
    def TemporalTransferFunction(self):
        def delayTransferFunction(freq,fSamp,delay):
            return np.exp(-2*complex(0,1)*freq*delay/fSamp)
        def wfsTransferFunction(freq,fSamp):
            return np.sinc(freq/fSamp/np.pi)*np.exp(-1*complex(0,1)*freq/fSamp)
        def servoTransferFunction(freq,fSamp,num,den):
            z = np.exp(-2*complex(0,1)*np.pi*freq/fSamp)
            ho_num = num[0] + num[1]*z + num[2]*z**2 + num[3]*z**3
            ho_den = 1 + den[0]*z + den[1]*z**2 + den[2]*z**3
            return ho_num/ho_den
        
        #1\ HIGH-ORDER LOOP
        self.trs.holoop.tf.freq = np.linspace(self.trs.holoop.freq/self.trs.wfs.nExp,self.trs.wfs.nExp/2,num=self.trs.wfs.nExp//2)*self.trs.holoop.freq/self.trs.wfs.nExp
        #1.2 wfs TF
        self.trs.holoop.tf.wfs  = wfsTransferFunction(self.trs.holoop.tf.freq,self.trs.holoop.freq)
        #1.3. Lag tf
        self.trs.holoop.tf.lag = delayTransferFunction(self.trs.holoop.tf.freq,self.trs.holoop.freq,self.trs.holoop.lat*self.trs.holoop.freq)
        #1.4 servo tf
        self.trs.holoop.tf.servo = servoTransferFunction(self.trs.holoop.tf.freq,self.trs.holoop.freq,self.trs.holoop.tf.num,self.trs.holoop.tf.den)
        #1.5. open loop tf
        self.trs.holoop.tf.ol = self.trs.holoop.tf.wfs*self.trs.holoop.tf.servo*self.trs.holoop.tf.lag
        #1.6. closed loop tf
        self.trs.holoop.tf.ctf = self.trs.holoop.tf.ol/(1+ self.trs.holoop.tf.ol)
        #1.7. rejection tf
        self.trs.holoop.tf.rtf = 1 - self.trs.holoop.tf.ctf
        #1.8. noise transfer function
        self.trs.holoop.tf.ntf = np.squeeze(self.trs.holoop.tf.servo/(1+self.trs.holoop.tf.ol))
        self.trs.holoop.tf.pn  = (np.trapz(self.trs.holoop.tf.freq,abs(self.trs.holoop.tf.ntf)**2)*2/self.trs.holoop.freq)
        
        # 2\ TT LOOP
        #2.1. Define the frequency vectors
        self.trs.ttloop.tf.freq = np.linspace(self.trs.ttloop.freq/self.trs.tipTilt.nExp,self.trs.tipTilt.nExp/2,self.trs.tipTilt.nExp//2)*self.trs.ttloop.freq/self.trs.tipTilt.nExp
        #2.2 wfs TF
        self.trs.ttloop.tf.wfs  = wfsTransferFunction(self.trs.ttloop.tf.freq,self.trs.ttloop.freq)
        #2.3 TT Lag tf
        self.trs.ttloop.tf.lag = delayTransferFunction(self.trs.ttloop.tf.freq,self.trs.ttloop.freq,self.trs.ttloop.lat*self.trs.ttloop.freq)
        #2.4 TT servo tf
        self.trs.ttloop.tf.servo = servoTransferFunction(self.trs.ttloop.tf.freq,self.trs.ttloop.freq,self.trs.ttloop.tf.num,self.trs.ttloop.tf.den)
        #2.5 open loop tf
        self.trs.ttloop.tf.ol  = self.trs.ttloop.tf.wfs*self.trs.ttloop.tf.servo*self.trs.ttloop.tf.lag
        #2.6 closed loop tf
        self.trs.ttloop.tf.ctf = self.trs.ttloop.tf.ol/(1+self.trs.ttloop.tf.ol)
        #2.7 rejection tf
        self.trs.ttloop.tf.rtf = 1 - self.trs.ttloop.tf.ctf
        #2.8 noise transfer function
        self.trs.ttloop.tf.ntf = np.squeeze(self.trs.ttloop.tf.servo/(1+self.trs.ttloop.tf.ol))
        self.trs.ttloop.tf.pn  = (np.trapz(self.trs.ttloop.tf.freq,abs(self.trs.ttloop.tf.ntf)**2)*2/self.trs.ttloop.freq)

    def GetNumberPhotons(self):
        """
        ESTIMATE THE NUMBER OF PHOTONS FROM WFS MAPS
        """
        nph = ron = nph_tt = ron_tt = 0
        
        nExp, nS, _ = self.trs.wfs.intensity.shape
        maps = self.trs.wfs.intensity.reshape((nExp,nS**2))
        if np.any(maps):
            pixInt = np.sort(maps/self.trs.wfs.gain ,axis=1)
            nSl_c  = np.count_nonzero(self.trs.mat.R[100,:,0])
            pixCog = pixInt[:,-nSl_c//2:]
            nph = np.mean(pixCog) # in photo-events/frame/subap            
            # Read-out noise estimation
            ron = np.median(np.std(pixCog,axis=0))
 
        
        if self.trs.aoMode == 'LGS':
              nph = np.mean(self.trs.tipTilt.intensity/self.trs.tipTilt.gain)
              ron = np.std(self.trs.tipTilt.intensity/self.trs.tipTilt.gain)
        else:
            nph_tt = nph
            ron_tt =ron
        
        return nph, ron, nph_tt, ron_tt
          
    def GetNoiseVariance(self,noiseMethod='autocorrelation',nshift=1,nfit=2):
        """
            ESTIMATE THE WFS NOISE VARIANCE
        """                    
        
        
        def SlopesToNoise(u,noiseMethod='autocorrelation',nshift=1,nfit=2,rtf=None):
        
            nF,nU      = u.shape
            u         -= np.mean(u,axis=0)
            Cnn        = np.zeros((nU,nU))
            validInput = np.argwhere(u.std(axis=0)!=0)
            
            if noiseMethod == 'rtf':
                
                fftS = fft.fft(u,axis=0)
                fftS = fftS[0:nF//2,:]/rtf
                # truncate to keep high-spatial frequencies only
                sub_dim = np.floor(9*nF/2/10)
                fftS = fftS[sub_dim:,:]
                cfftS= np.conj(fftS)
       
                for i in range(nU): 
                    if validInput[i]:
                        # Get the cross PSD
                        crossPSD  = fftS[i,:] * cfftS[i:,:]
                        # Estimate the noise plateau
                        Cnn[i,i:]  = np.median(np.real(crossPSD),axis=0)
                    
                Cnn = (np.transpose(Cnn) + Cnn - np.diag(np.diag(Cnn)))//nF
                
            if noiseMethod == 'interpolation':
                # Polynomial fitting procedure
                delay   = np.linspace(0,1,nfit+1)
                for i in np.arange(0,nU,1):
                    g      = fft.ifft(fft.fft(u[i,:])*np.conjugate(fft.fft(u[i,:])))/nF
                    mx     = g.max()
                    pfit   = np.polyfit(delay[1:nfit+1],g[1:nfit+1],nfit)
                    yfit   = np.polyval(pfit,delay)
                    Cnn[i,i] = mx - yfit[0]              
        
            if noiseMethod == 'autocorrelation':
                du_n  = u - np.roll(u,-nshift,axis=0)
                du_p  = u - np.roll(u,nshift,axis=0)
                Cnn   = 0.5*(np.matmul(u.T,du_n + du_p))/nF
            
            return Cnn
            
        if noiseMethod == 'nonoise':
            Cn_ao = 0
            Cn_tt = 0
            
        else:
            rtf  = self.trs.holoop.tf.ctf/self.trs.holoop.tf.wfs
            Cn_ao = SlopesToNoise(self.trs.dm.com,noiseMethod=noiseMethod,nshift=nshift,nfit=nfit,rtf=rtf)
            if self.trs.aoMode == 'LGS':
                rtf  = self.trs.ttloop.tf.ctf/self.trs.ttloop.tf.wfs
                Cn_tt = SlopesToNoise(self.trs.tipTilt.com,noiseMethod=noiseMethod,nshift=nshift,nfit=nfit)
            else:
                Cn_tt = 0
                
        return Cn_ao, Cn_tt
     
#%%    
    def ReconstructZernike(self,nZer=None,wfsMask=None,j0=4):
        """
        Reconstrut the covariance matrix of the Zernike coefficients from the reconstructed open-loop wavefront
        """
        # defining Zernike modes
        if nZer == None:
            nZer = int(0.6*self.trs.dm.validActuators.sum())
            
        self.trs.wfs.jIndex = list(range(j0,nZer+j0))
            
        # defining the pupil mask and the Zernike modes
        if wfsMask==None:
            wfsMask = fits.getdata(self.trs.tel.path_pupil)
            ang     = self.trs.wfs.theta[0] + self.trs.tel.pupilAngle
            wfsMask = FourierUtils.interpolateSupport(rotate(wfsMask,ang,reshape=False),self.trs.tel.resolution).astype(bool)
        
        self.z = zernike(self.trs.wfs.jIndex,self.trs.tel.resolution,pupil=wfsMask)
        
        # computing the Zernike reconstructor
        u2ph = self.trs.mat.dmIF
        zM   = self.z.modes.T.reshape((self.trs.tel.resolution**2,nZer))
        ph2z = np.dot(np.linalg.pinv(np.dot(zM.T,zM)),zM.T)
        self.trs.mat.u2z = np.dot(ph2z,u2ph)
        
        # open-loop reconstruction
        dt    = self.trs.holoop.lat * self.trs.holoop.freq
        dt_up = int(np.ceil(dt))
        dt_lo = int(np.floor(dt))
        self.trs.dm.com_delta = (1-dt_up+dt)*np.roll(self.trs.rec.res,(-dt_up,0)) + (1-dt+dt_lo)*np.roll(self.trs.rec.res,(-dt_lo,0))
        self.trs.dm.u_ol      = self.trs.dm.com + self.trs.dm.com_delta
        self.trs.dm.u_ol     -= np.mean(self.trs.dm.u_ol,axis=1)[:,np.newaxis]
        
        # reconstructing the amplitude of Zernike coefficients
        self.trs.wfs.coeffs = np.dot(self.trs.mat.u2z,self.trs.dm.u_ol.T)
        self.trs.wfs.coeffs -= np.mean(self.trs.wfs.coeffs,axis=1)[:,np.newaxis]
        Cz_ho  = np.dot(self.trs.wfs.coeffs,self.trs.wfs.coeffs.T)/self.trs.wfs.nExp
        
        #tip-tilt case
        if self.trs.aoMode == 'LGS':
            Cz_tt = np.matmul(self.trs.tipTilt.com.T,self.trs.tipTilt.com)/self.trs.tipTilt.nExp
        else:
            Cz_tt = Cz_ho[0:1,0:1]
            
        return Cz_ho, Cz_tt
    
    def GetAtmosphereStatistics(self,ftol=1e-5,xtol=1e-5,gtol=1e-5,max_nfev=100,\
                                noise=1,quantile=0.95,nWin=10,verbose=2):
        
        # ---- DEFINING THE COST FUNCTIONS
        z = self.z
        D = self.trs.tel.D
        class CostClass(object):
            def __init__(self):
                self.iter = 0
            def __call__(self,y):
                if (self.iter%3)==0 and (verbose == 0 or verbose == 1): print("-",end="")
                self.iter += 1
                var_mod = z.CoefficientsVariance(D/y)
                return (var_mod - var_emp)
        cost = CostClass()
    
        # ---- GETTING THE INPUT : VARIANCE OF ZERNIKE MODE
        self.trs.wfs.var_meas      = np.diag(self.trs.wfs.Cz_ao)
        self.trs.wfs.var_noise_zer = np.diag(np.dot(np.dot(self.trs.mat.u2z,self.trs.wfs.Cn_ao),self.trs.mat.u2z.T))
        var_emp                    = (2*np.pi/self.trs.atm.wvl)**2 *(self.trs.wfs.var_meas - noise*self.trs.wfs.var_noise_zer)
        
        # ---- DATA-FITTING WITH A LEVENBERG-MARQUARDT ALGORITHM
        # minimization
        x0  = np.array([0.2,self.trs.atm.L0])
        res = least_squares(cost,x0,method='lm',ftol=ftol, xtol=xtol, gtol=gtol,\
                            max_nfev=max_nfev,verbose=max(verbose,0))
        
        # fit uncertainties
        res.xerr   = confidence_interval(res.fun,res.jac)
        
        # unpacking results
        r0 = res.x[0] # r0 los at the atm wavelength
        L0 = res.x[1]
        
        # ---- MEASURING THE COHERENCE TIME
        # measuring the windspeed
        v0, dv0  = self.GetWindSpeed(noise=noise,quantile=quantile,nWin=nWin)
        # computing the coherence time (Roddier+81)
        tau0 = 0.314 * r0/v0
        
        # ---- COMPUTING UNCERTAINTIES
        dr0   = res.xerr[0]
        dL0   = res.xerr[1]
        dtau0 = 0.314*np.hypot(dr0/v0,r0*dv0/v0**2)
        
        return r0, L0, tau0, v0 , dr0, dL0, dtau0, dv0
    
    def GetWindSpeed(self,coeff=1.56,noise=1,nWin=10,quantile=0.95):
        
        
        def GetQuantilesStudent(quantile=0.95,nWin=10):
            # quantiles
            q   = np.array([0.75,0.8,0.85,0.9,0.95,0.975,0.99,0.995,0.997,0.998] )
            idq = q.searchsorted(quantile)
            
            # values of the Students pdf
            t1 = [1,1.376,1.963,3.078,6.31,12.71,31.82,63.66,106.1,159.2]
            t2 = [0.816,1.061,1.386,1.886,2.92,4.303,6.965,9.925,12.85,15.76]
            t3 = [0.765,0.978,1.25,1.638,2.353,3.182,4.541,5.841,6.994,8.053] 
            t4 = [0.741,0.941,1.19,1.533,2.132,2.776,3.747,4.604,5.321,5.951] 
            t5 = [0.727,0.92,1.156,1.476,2.015,2.571,3.365,4.032,4.57,5.03] 
            t6 = [0.718,0.906,1.134,1.44,1.943,2.447,3.143,3.707,4.152,4.524] 
            t7 = [0.727,0.92,1.156,1.476,2.015,2.571,3.365,4.032,4.57,5.03] 
            t8 = [0.706,0.889,1.108,1.397,1.86,2.306,2.896,3.355,3.705,3.991] 
            t9 = [0.703,0.883,1.1,1.383,1.833,2.262,2.821,3.25,3.573,3.835]
            t10 = [0.7,0.879,1.093,1.372,1.812,2.228,2.764,3.169,3.472,3.716] 
            area  = np.array([t1,t2,t3,t4,t5,t6,t7,t8,t9,t10])
            
            # select the value
            if nWin > 10:
                print('Warning : the pdf for more than 7 degrees of freedom is not implemented. Stick to 7')
                n = min(7,nWin)
            else:
                n = nWin
                
            return area[n-1,idq]
        
        # ---- DERIVING THE AUTOCORRELATION OF ZERNIKE MODES
        z    = self.trs.wfs.coeffs
        z_pad= np.pad(z,((0,0),(0,z.shape[1])))
        fftZ = fft.fft(z_pad,axis=1)
        cov  = fft.ifft(abs(fftZ)**2,axis=1)/z.shape[1]
        cov  = np.real(cov[:,:cov.shape[1]//2])
        
        # ---- DENOISING
        cov -=  noise*self.trs.wfs.var_noise_zer[:,np.newaxis]
        
        # ---- MEASURING THE 1/e WIDTH
        
        # normalizing max the max
        cov0 = cov[:,0]
        cov /= cov0[:,np.newaxis]
        thres = 1/np.exp(1)
        
        # loop on each mode
        nZ   = cov.shape[0]
        tau  = np.zeros(nZ)
        dtau = np.zeros(nZ)
        t5  = GetQuantilesStudent(quantile=quantile,nWin=nWin)
        for i in range(nZ):
            # get the upper bound
            indlim = np.argwhere(cov[i] < thres)[0][0]
            # defining the window for the delay and get the corresponding autocorrelation function
            x   = np.linspace(indlim-nWin//2,indlim+nWin//2,num=nWin).astype(int)
            y   = cov[i,x]
            # computing moments over the window
            Sx  = x.sum()
            Sxx = np.sum(x**2)
            Sy  = y.sum()
            Syy = np.sum(y**2)      
            Sxy = np.sum(x*y)
            # the slope over the window
            slope  = (nWin*Sxy - Sx*Sy)/(nWin*Sxx - Sx**2)
            offset = Sy/nWin - slope*Sx/nWin
            tau[i] = (thres - offset)/slope/self.trs.holoop.freq
            # 95 confidence interval for the slope and offset estimates
            de = np.sqrt(1/(nWin*(nWin-2)) * (nWin*Syy - Sy**2 - slope**2 * (nWin*Sxx - Sx**2)))
            db = t5*de * np.sqrt( nWin/(nWin*Sxx - Sx**2))
            da = db * np.sqrt(Sxx/ nWin)
            # uncertainty of the tau measurements
            dtau[i] = np.hypot(da/slope,db*(thres - offset)/slope**2)/self.trs.holoop.freq
            
        # ---- AVERAGING OVER RADIAL MODES
        nrad = self.z.n
        n_unique = np.unique(nrad)
        nu = len(n_unique)
        taun = np.zeros(nu)
        dtaun= np.zeros(nu)
        for i in range(nu):
            taun[i] = np.mean(tau[nrad == n_unique[i]])
            dtaun[i] =  np.mean(dtau[nrad == n_unique[i]])
        # ---- MEASURING WINDSPEED (CONAN95 + FUSCO+04)
        #Note : there's a mistake in Fusco+04 : v = D*sum1/(coeff*pi*sum2)
        sum1 =  ((n_unique + 1)/taun).sum()
        sum2 = ((n_unique + 1)**2).sum()
        v0   = self.trs.tel.D * sum1 / (coeff*np.pi*0.3*sum2)
        
        # ---- COMPUTING UNCERTAINTIES
        dsum1 = (dtaun*(n_unique + 1)/taun**2).sum()
        dv0   = self.trs.tel.D * dsum1 / (coeff*np.pi*0.3*sum2)
        
        return v0, dv0