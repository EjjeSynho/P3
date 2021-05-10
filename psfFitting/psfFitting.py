#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:33:19 2021

@author: omartin
"""

#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
from scipy.optimize import least_squares
import aoSystem.FourierUtils as FourierUtils
from psfFitting.confidenceInterval import confidence_interval
import matplotlib.pyplot as plt

#%%
def psfFitting(image,psfModelInst,x0,weights=None,fixed=None,method='trf',normType=1,
               bounds=None,ftol=1e-8,xtol=1e-8,gtol=1e-8,max_nfev=1000,verbose=0):
    """Fit a PSF with a parametric model solving the least-square problem
       epsilon(x) = SUM_pixel { weights * (amp * Model(x) + bck - psf)² }
    
    Parameters
    ----------
    image : numpy.ndarray
        The experimental image to be fitted
    Model : class
        The class representing the PSF model
    x0 : tuple, list, numpy.ndarray
        Initial guess for parameters
    weights : numpy.ndarray
        Least-square weighting matrix (same size as `image`)
        Inverse of the noise's variance
        Default: uniform weighting
    fixed : numpy.ndarray
        Fix some parameters to their initial value (default: None)
    method : str or callable, optional : trf, dogbox or lm
    ftol, xtol, gtol:float, optional
        Tolerance for termination on the funcion, inputs and gradient. For detailed control, use solver-specific options.
    max_nfev : int, optionnal
        Maximal number of function evaluation
    verbose : int, optional
        Print information about the fitting during the minimization
            0 : progress bar only
            1: number of iteration
            2: minimization details for each iteration
        
    Returns
    -------
    out.x : numpy.array
            Parameters at optimum
       .dxdy : tuple of 2 floats
           PSF shift at optimum
       .flux_bck : tuple of two floats
           Estimated image flux and background
       .psf : numpy.ndarray (dim=2)
           Image of the PSF model at optimum
       .success : bool
           Minimization success
       .status : int
           Minimization status (see scipy doc)
       .message : string
           Human readable minimization status
       .active_mask : numpy.array
           Saturated bounds
       .nfev : int
           Number of function evaluations
       .cost : float
           Value of cost function at optimum
    """
    
    # WEIGHTS
    if weights is None: weights = np.ones_like(image)
    elif len(image)!=len(weights): raise ValueError("Keyword `weights` must have same number of elements as `psf`")
    sqW = np.sqrt(weights)
    
    # NORMALIZING THE IMAGE
    im_norm,param = FourierUtils.normalizeImage(image,normType=normType)
    
    # DEFINING THE COST FUNCTIONS
    class CostClass(object):
        def __init__(self):
            self.iter = 0
        def __call__(self,y):
            if (self.iter%3)==0 and (method=='lm' or verbose == 0 or verbose == 1): print("-",end="")
            self.iter += 1
            im_est = np.squeeze(psfModelInst(mini2input(y)))
            return (sqW * (im_est - im_norm)).reshape(-1)
    cost = CostClass()   
    
    # DEFINING THE BOUNDS
    if fixed is not None:
        if len(fixed)!=len(x0): raise ValueError("When defined, `fixed` must be same size as `x0`")
        FREE    = [not fixed[i] for i in range(len(fixed))]
        INDFREE = np.where(FREE)[0]
        
    def get_bounds(inst):
        if bounds is None:
            b_low = inst.bounds[0]
            if fixed is not None: b_low = np.take(b_low,INDFREE)
            b_up = inst.bounds[1]
            if fixed is not None: b_up = np.take(b_up,INDFREE)
            return (b_low,b_up)
        else:
            return (np.take(bounds[0],INDFREE),np.take(bounds[1],INDFREE))
    
    def input2mini(x):
        # Transform user parameters to minimizer parameters
        if fixed is None: xfree = x
        else: xfree = np.take(x,INDFREE)
        return xfree
    
    def mini2input(y,forceZero=False):
        # Transform minimizer parameters to user parameters
        if fixed is None:
            xall = y
        else:
            if forceZero:
                xall = np.zeros_like(x0)
            else:
                xall = np.copy(x0)
            for i in range(len(y)):
                xall[INDFREE[i]] = y[i]
        return xall
            
    # PERFORMING MINIMIZATION WITH CONSTRAINS AND BOUNDS
    if method == 'trf':
        result = least_squares(cost,input2mini(x0),method='trf',bounds=get_bounds(psfModelInst),
                               ftol=ftol, xtol=xtol, gtol=gtol,max_nfev=max_nfev,verbose=max(verbose,0))
    else:
        result = least_squares(cost,input2mini(x0),method='lm',
                               ftol=ftol, xtol=xtol, gtol=gtol,max_nfev=max_nfev,verbose=max(verbose,0))

    # update parameters
    result.x      = mini2input(result.x)
    result.xinit  = x0
    result.im_sky = image
    # scale fitted image
    result.im_fit = FourierUtils.normalizeImage(np.squeeze(psfModelInst(result.x)),param=param,normType=normType)
    # psf
    xpsf          = np.copy(result.x)
    nparam        = len(result.x) - 3*psfModelInst.ao.src.nSrc - 1
    if nparam > psfModelInst.ao.tel.nModes:
        nparam -= psfModelInst.ao.tel.nModes
        
    xpsf[nparam]  = 1.0 # flux=1
    xpsf[nparam+1:nparam+3*psfModelInst.ao.src.nSrc+1]   = 0.0 # dx,dy,bkcg=0
    result.psf    = np.squeeze(psfModelInst(xpsf))
    result        = evaluateFittingQuality(result,psfModelInst)
    
    # static map
    nModes = psfModelInst.ao.tel.nModes
    if (nModes) > 0 and len(result.x) > nModes + psfModelInst.ao.src.nSrc:
        result.opd = (psfModelInst.ao.tel.statModes*result.x[-nModes:]).sum(axis=2)
    
    # 95% confidence interval
    try:
        result.xerr = mini2input(confidence_interval(result.fun,result.jac),forceZero=True)
    except:
        print('Identification of the confidence interval failed ')
        result.xerr = []
    
    return result


def evaluateFittingQuality(result,psfModelInst):
    
    # ESTIMATING IMAGE-BASED METRICS
    def meanErrors(sky,fit):
        mse = 1e2*np.sqrt(np.sum((sky-fit)**2))/sky.sum()
        mae = 1e2*np.sum(abs(sky-fit))/sky.sum()
        fvu = 1e2*np.sum((sky-fit)**2)/np.sum((sky-sky.mean())**2)
        return mse,mae,fvu
    
    result.SR_sky   = FourierUtils.getStrehl(result.im_sky,psfModelInst.ao.tel.pupil,psfModelInst.freq.sampCen)
    result.SR_fit   = FourierUtils.getStrehl(result.im_fit,psfModelInst.ao.tel.pupil,psfModelInst.freq.sampCen)
    result.FWHMx_sky , result.FWHMy_sky = FourierUtils.getFWHM(result.im_sky,psfModelInst.ao.cam.psInMas,nargout=2)
    result.FWHMx_fit , result.FWHMy_fit = FourierUtils.getFWHM(result.im_fit,psfModelInst.ao.cam.psInMas,nargout=2)
    result.mse, result.mae , result.fvu = meanErrors(result.im_sky,result.im_fit)
    
    return result

def displayResults(psfModelInst,res,vmin=None,vmax=None,nBox=None,scale='log10abs'):
    """
        Displaying PSF and key metrics
    """
    
    # RETRIEVING LABELS
    psfrLabel = psfModelInst.tag
    instLabel = psfModelInst.ao.cam.tag
    
    # MANAGING THE IMAGE SIZE TO BE DIPSLAYED
    nPix  = res.im_sky.shape[0]
    if nBox is not None and nBox < nPix:
        nCrop = nPix/nBox
        im_sky = FourierUtils.cropSupport(res.im_sky,nCrop)
        im_fit = FourierUtils.cropSupport(res.im_fit,nCrop)
    else:
        im_sky = res.im_sky
        im_fit = res.im_fit
    
    im_diff= im_fit - im_sky
    
    # DEFINING THE FUNCTION TO APPLY
    if scale == 'log10abs':
        fun = lambda x: np.log10(abs(x))
    elif scale == 'arcsinh':
        fun = lambda x: np.arcsinh(x)
    elif scale == 'linear':
        fun = lambda x: x
    else:
        scale = 'linear'
        print('Scale input is not recognized, go linear')
        fun = lambda x: x
            
    # MANAGING THE IMAGE RANGE
    if vmin is None:
        if scale =='log10abs':
            vmin = 0
        else:
            vmin = np.min( [np.min(fun(im_sky)), np.min(fun(im_fit))])
    if vmax is None:
        vmax = np.max([np.max(fun(im_sky)), np.max(fun(im_fit))])
    
    # IMAGES
    plt.figure()
    plt.subplot(231)
    plt.imshow(fun(im_sky),vmin=vmin,vmax=vmax)
    plt.title(instLabel)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(fun(im_fit),vmin=vmin,vmax=vmax)
    plt.title(psfrLabel)
    plt.colorbar()
    plt.axis('off')  
    plt.subplot(233)
    plt.imshow(fun(im_diff),vmin=vmin,vmax=vmax)
    plt.title(psfrLabel+' - ' + instLabel)
    plt.axis('off')
    plt.colorbar()
    
    # AZIMUTHAL AVERAGES
    x,prof_sky = FourierUtils.radial_profile(im_sky,pixelscale=psfModelInst.ao.cam.psInMas)
    prof_fit   = FourierUtils.radial_profile(im_fit,nargout=1)    
    plt.subplot(212)
    plt.plot(x,fun(prof_sky),label=instLabel)
    plt.plot(x,fun(prof_fit),label=psfrLabel)
    
    #plt.ylim([vmin,vmax])
    plt.legend()
    plt.ylabel('Azimuthal profile ('+ scale +'-scale)')
    plt.xlabel('Distance from on-axis [mas]')