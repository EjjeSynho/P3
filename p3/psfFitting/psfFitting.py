#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:33:19 2021

@author: omartin
"""

import numpy as np
from scipy.optimize import least_squares
import aoSystem.FourierUtils as FourierUtils
from psfFitting.confidenceInterval import confidence_interval
from psfFitting.imageModel import imageModel

from copy import deepcopy
import matplotlib
from matplotlib import colors, scale
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


#%%
def psfFitting(image, psfModelInst, x0, fixed=None, weights=None, method='trf',
               spectralStacking=True, spatialStacking=True, normType=1,
               bounds=None, ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=1000,
               verbose=0, alpha_positivity=None):

    """Fit a PSF with a parametric model solving the least-square problem
       epsilon(x) = SUM_pixel { weights * (amp * Model(x) + bck - psf)² }

    Parameters
    ----------
    image : numpy.ndarray
        The experimental image to be fitted
    Model : class
        The class representing the PSF model
    x0 : dictionary
        Initial guess for parameters
    #########x0 : tuple, list, numpy.ndarray
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

    # Weights
    if weights is None: weights = np.ones_like(image)
    elif len(image) != len(weights): raise ValueError("Keyword `weights` must have same number of elements as `psf`")
    sqW = np.sqrt(weights)

    # Normalizing the image
    if image.min() < 0:
        image -= image.min()
    im_norm, param = FourierUtils.normalizeImage(image, normType=normType)
    nPix = im_norm.shape[1]

    # Defining the cost functions
    class CostClass(object):
        def __init__(self, alpha_positivity=None):
            self.iter = 0
            self.alpha_positivity = alpha_positivity
            
        def regularization_positivity(self, im_res):
            R = np.ones_like(im_res)
            R[im_res>0] = 0
            return self.alpha_positivity*R

        def __call__(self, y):
            if (self.iter%3)==0 and (method=='lm' or verbose == 0 or verbose == 1): print("-", end="")
            self.iter += 1
            # image model
            im_est = imageModel(psfModelInst(mini2input(y),
                                nPix = im_norm.shape[0]),
                                spatialStacking = spatialStacking, spectralStacking = spectralStacking,
                                saturation = psfModelInst.ao.cam.saturation/param)

            im_res = im_norm - im_est
            df_term = sqW * im_res
            if alpha_positivity is not None:
                #im_in = medfilt2d(im_res)
                df_term *= np.sqrt(1 + self.regularization_positivity(im_res))

            return df_term.reshape(-1)

    cost = CostClass(alpha_positivity=alpha_positivity)

    # DEFINING THE BOUNDS
    if fixed is not None:
        dicts_equal = bool(np.prod([item in fixed for item in x0])) # if only 1 element=False -> prod=0
        if not dicts_equal: #len(fixed) != len(x0):
            #raise ValueError("When defined, `fixed` must be same size as `x0`")
            raise ValueError("When defined, `fixed` must have same fields as `x0`")
        #INDFREE = np.where( [not fixed[i] for i in range(len(fixed))] )[0]
    optimized_items = [item for item in x0 if fixed[item] == False]

    if len(optimized_items) == 0:
        raise ValueError('Error: nothing to optimize! All parameters are fixed')

    #INDFREE = [0,1,2,3,4,5,6]

    def get_bounds(inst):
        # This is done to preserve the order of dict's fields the same as for x0 
        upper_bounds = deepcopy(x0)
        lower_bounds = deepcopy(x0)

        for item in upper_bounds: # Fill for all fittable but unspecified params
            lower_bounds[item] = -np.Inf
            upper_bounds[item] =  np.Inf
            if item not in lower_bounds:
                raise ValueError('Error: content of upper- and lower- bounds should match!')
                
        for item in inst.bounds[0]: # Fill for all fittable and specified params
            lower_bounds[item] = inst.bounds[0][item]
            upper_bounds[item] = inst.bounds[1][item]

        #TODO: multiple sources support
        lower_bounds['jitterX'] = 0.
        lower_bounds['jitterY'] = 0.
        lower_bounds['Jxy'] = 0.
        lower_bounds['F'] = 0.
        lower_bounds['bkg'] = 0.
        
        upper_bounds['F'] = 1.

        return input2mini(lower_bounds), input2mini(upper_bounds)

        '''
        b_low = inst.bounds[0]
        b_up  = inst.bounds[1]
        if bounds == None:
            b_low = inst.bounds[0]
            b_up  = inst.bounds[1]
            if fixed is not None:
                b_low = np.take(b_low, INDFREE)
                b_up  = np.take(b_up,  INDFREE)
            return (b_low, b_up)
        else:
            return ( np.take(bounds[0],INDFREE), np.take(bounds[1],INDFREE) )
        '''

    def input2mini(x):  #TODO: support multiple sources
        # Transform user parameters to minimizer parameters        
        #TODO: manage arrays for stellar params or with list params in general
        return np.array( [x[item] for item in optimized_items] )
        #if fixed is None: xfree = x
        #else: xfree = np.take(x, INDFREE)
        #return xfree

    def mini2input(y, forceZero=False): #TODO: support multiple sources
        # Transform minimizer parameters to user parameters
        xall = deepcopy(x0)
        if forceZero:
            for item in xall:
                xall[item] = 0.0
        for i,item in enumerate(optimized_items):
            xall[item] = y[i]
        return xall
        '''
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
        '''

    # PERFORMING MINIMIZATION WITH CONSTRAINS AND BOUNDS
    if method == 'trf':
        result = least_squares(cost,
                               input2mini(x0),
                               method = 'trf',
                               bounds = get_bounds(psfModelInst),
                               ftol=ftol, xtol=xtol, gtol=gtol,
                               max_nfev = max_nfev,
                               verbose = max(verbose, 0),
                               loss = "linear")
    else:
        result = least_squares(cost,
                               input2mini(x0),
                               method = 'lm',
                               ftol=ftol, xtol=xtol, gtol=gtol,
                               max_nfev = max_nfev,
                               verbose = max(verbose, 0))

    # update parameters
    result.x = mini2input(result.x)
    result.xinit  = x0
    result.im_sky = image

    # scale fitted image
    tmp = imageModel(psfModelInst(result.x, nPix=nPix),
                     spatialStacking  = spatialStacking,
                     spectralStacking = spectralStacking,
                     saturation = psfModelInst.ao.cam.saturation/param)
                     
    result.im_fit = FourierUtils.normalizeImage(tmp, param=param, normType=normType)
    result.im_dif = result.im_sky - result.im_fit
    
    # PSF
    xpsf = deepcopy(result.x)
    #n_src = psfModelInst.ao.src.nSrc
    #id_flux = psfModelInst.n_param_atm + psfModelInst.n_param_dphi + 3
    #xpsf[id_flux] = 1.0 # flux=1
    #xpsf[id_flux+1:id_flux+3*n_src+1] = 0.0 # dx,dy,bkcg=0
    result.psf = np.squeeze(psfModelInst(xpsf, nPix=nPix))
    result = evaluateFittingQuality(result,psfModelInst) #TODO: support multiple sources

    # static map
    nModes = psfModelInst.ao.tel.nModes #TODO: support static
    if (nModes) > 0 and len(result.x) > nModes + psfModelInst.ao.src.nSrc:
        result.opd = (psfModelInst.ao.tel.statModes*result.x[-nModes:]).sum(axis=2)

    # 95% confidence interval
    try:
        result.xerr = mini2input(confidence_interval(result.fun, result.jac), forceZero=True)
    except:
        print('Identification of the confidence interval failed ')
        if hasattr(result.x, '__iter__'):
            result.xerr = list(-1 * np.ones(len(result.x)))
        else:
            result.xerr = [-1]

    return result


def psfEstimated(x0, psfModelInst, im, weights=1):
    _, param = FourierUtils.normalizeImage(im, normType=1)
    im_est = imageModel(psfModelInst(x0, nPix=im.shape[0]), spatialStacking=True, spectralStacking=True, saturation=psfModelInst.ao.cam.saturation/param)
    return im_est*weights


def evaluateFittingQuality(result, psfModelInst):
    # estimating image-based metrics
    def meanErrors(sky,fit):
        mse = 1e2*np.sqrt(np.sum((sky-fit)**2))/sky.sum()
        mae = 1e2*np.sum(abs(sky-fit))/sky.sum()
        fvu = 1e2*np.sum((sky-fit)**2)/np.sum((sky-sky.mean())**2)
        return mse,mae,fvu

    if np.ndim(result.im_sky) == 2:
        # case fit of a 2D image
        result.SR_sky = FourierUtils.getStrehl(result.im_sky, psfModelInst.ao.tel.pupil, psfModelInst.freq.sampRef)
        result.SR_fit = FourierUtils.getStrehl(result.im_fit, psfModelInst.ao.tel.pupil, psfModelInst.freq.sampRef)
        result.FWHMx_sky, result.FWHMy_sky = FourierUtils.getFWHM(result.im_sky, psfModelInst.ao.cam.psInMas, nargout=2)
        result.FWHMx_fit, result.FWHMy_fit = FourierUtils.getFWHM(result.im_fit, psfModelInst.ao.cam.psInMas, nargout=2)
        result.mse, result.mae, result.fvu = meanErrors(result.im_sky, result.im_fit)

    elif (np.ndim(result.im_sky) == 3) and (psfModelInst.nwvl>1):
        # case fit of an hyperspectral data cube
        nWvl = psfModelInst.nwvl
        result.SR_sky = result.SR_fit = np.zeros(nWvl)
        result.FWHMx_sky = result.FWHMy_sky = np.zeros(nWvl)
        result.FWHMx_fit = result.FWHMy_fit = np.zeros(nWvl)
        result.mse = result.mae = result.fvu= np.zeros(nWvl)

        for j in range(nWvl):
            ims_j = result.im_sky[:,:,j]
            imf_j = result.im_fit[:,:,j]
            result.SR_sky[j] = FourierUtils.getStrehl(ims_j,psfModelInst.ao.tel.pupil,psfModelInst.freq[j].samp[0])
            result.SR_fit[j] = FourierUtils.getStrehl(imf_j,psfModelInst.ao.tel.pupil,psfModelInst.freq[j].samp[0])
            result.FWHMx_sky[j] , result.FWHMy_sky[j] = FourierUtils.getFWHM(ims_j,psfModelInst.freq[j].psInMas[0],nargout=2)
            result.FWHMx_fit[j] , result.FWHMy_fit[j] = FourierUtils.getFWHM(imf_j,psfModelInst.freq[j].psInMas[0],nargout=2)
            result.mse[j], result.mae[j] , result.fvu[j] = meanErrors(ims_j,imf_j)
    return result


def displayResults(psfModelInst, res, vmin=None, vmax=None, vminc=None, vmaxc=None, nBox=None,
                   scale='log10abs', figsize=(11,10), show=True, title='', y_label=''):
    """
        Displaying PSF and key metrics
    """
    # retrieving labels
    psfrLabel = psfModelInst.tag
    instLabel = psfModelInst.ao.cam.tag

    # managing the image size to be displayed
    nPix  = res.im_sky.shape[0]
    if nBox != None and nBox < nPix:
        nCrop = nPix/nBox
        im_sky = FourierUtils.cropSupport(res.im_sky, nCrop)
        im_fit = FourierUtils.cropSupport(res.im_fit, nCrop)
    else:
        im_sky = res.im_sky
        im_fit = res.im_fit

    im_diff = np.abs(im_sky - im_fit)

    if vmax == None:
        vmax = np.max([np.max(im_sky), np.max(im_fit)])
    if vmin == None:
        vmin = np.max([np.min(im_sky), np.min(im_fit)]) * int(scale=='log10abs')
   
    def forward(x):
        return np.arcsinh(x)
    def inverse(x):
        return np.sin(x)

    # DEFINING THE FUNCTION TO APPLY
    if   scale == 'log10abs':
        norm=colors.LogNorm(vmin=np.abs(vmin), vmax=vmax)
        im_fit = np.abs(im_fit)
        im_sky = np.abs(im_sky)
    elif scale == 'arcsinh':  norm=colors.FuncNorm((forward, inverse), vmin=vmin, vmax=vmax)
    elif scale == 'linear':   norm=colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        scale = 'linear'
        print('Scale input is not recognized, go linear')
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Images
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='darkslateblue')
 
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(2,3,1)
    im1 = ax1.imshow(im_sky, norm=norm)
    ax1.title.set_text(instLabel)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='10%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    ax1.axis('off')

    ax2 = plt.subplot(2,3,2)
    im2 = ax2.imshow(im_fit, norm=norm)
    ax2.title.set_text(psfrLabel)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='10%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    ax2.axis('off')

    ax3 = plt.subplot(2,3,3)
    im3 = ax3.imshow(im_diff, norm=norm)
    ax3.title.set_text('Abs. difference')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='10%', pad=0.05)
    cbar = fig.colorbar(im3, cax=cax, orientation='vertical')
    ax3.axis('off')
    cbar.set_label(y_label, labelpad=10)

    # Azimuthal averages
    x,prof_sky = FourierUtils.radial_profile(im_sky, pixelscale=psfModelInst.ao.cam.psInMas)
    prof_fit   = FourierUtils.radial_profile(im_fit, nargout=1)
    diff_profile = np.abs(prof_fit-prof_sky)/prof_sky.max()*100 # percents
    
    ax4 = plt.subplot(2,1,2)
    if   scale == 'log10abs': plt.yscale('log')
    elif scale == 'arcsinh':  plt.yscale('function', functions=(forward, inverse))
    elif scale == 'linear':   plt.yscale('linear')
    else: plt.yscale('linear')
    l1 = ax4.plot(x, prof_sky, label=instLabel)
    l2 = ax4.plot(x, prof_fit, label=psfrLabel)
    ax5 = ax4.twinx()
    l3 = ax5.plot(x, diff_profile, label='Difference', color='green')
    ax5.set_ylabel('Relative difference, [%]', labelpad=10)
    ax5.set_ylim([0, diff_profile.max()*1.5])
    ax5.set_xlim([x.min(), x.max()])

    # added these three lines
    ls = l1+l2+l3
    labs = [l.get_label() for l in ls]
    ax4.legend(ls, labs, loc=0)
    ax4.title.set_text('Azimuthal profile (in '+scale+' scale)'+title)
    ax4.set_ylabel(r'Intensity, $[F_{\lambda}/10^{-20}ergs...etc]$')
    ax4.set_xlabel('Distance from on-axis, [mas]', labelpad=5)
    ax4.set_xlim([x.min(), x.max()])
    
    plt.subplots_adjust(hspace=-0.35)

    if vminc is not None and vmaxc is not None:
        ax4.set_ylim([vminc, vmaxc])
    ax4.grid()

    fig.tight_layout()
    if show: plt.show()


def psfEvaluate(image, psfModelInst, x0, weights=None,
               spectralStacking=True, spatialStacking=True, normType=1):
    # Weights
    if weights is None: weights = np.ones_like(image)
    elif len(image) != len(weights): raise ValueError("Keyword `weights` must have same number of elements as `psf`")

    # Normalizing the image
    if image.min() < 0:
        image -= image.min()
    im_norm, param = FourierUtils.normalizeImage(image, normType=normType)
    nPix = im_norm.shape[1]

    class EvalResult:
        def __init__(self) -> None:
            self.x = None
            self.psf = None
            self.im_sky = None
            self.im_fit = None
            self.im_dif = None
            
    result = EvalResult()

    # update parameters
    result.x = deepcopy(x0)
    result.im_sky = image

    # scale fitted image
    tmp = imageModel(psfModelInst(result.x, nPix=nPix),
                     spatialStacking  = spatialStacking,
                     spectralStacking = spectralStacking,
                     saturation = psfModelInst.ao.cam.saturation/param)
                     
    result.im_fit = FourierUtils.normalizeImage(tmp, param=param, normType=normType)
    result.im_dif = result.im_sky - result.im_fit
    
    if weights is not None:
        result.im_fit *= weights
        result.im_sky *= weights
        result.im_dif *= weights

    xpsf = deepcopy(result.x)
    result.psf = np.squeeze(psfModelInst(xpsf, nPix=nPix))
    result = evaluateFittingQuality(result, psfModelInst) #TODO: support multiple sources
    return result

# %%
