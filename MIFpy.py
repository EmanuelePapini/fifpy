"""
 Iterative Filtering python package

 Dependencies : numpy, scipy, numba, joblib  //TO CHECK

 This is a full rewriting of MIF, not a wrapper of the original matlab code by Cicone
 
 Authors: 
    Python version: Emanuele Papini - INAF (emanuele.papini@inaf.it) 

"""




import os
import numpy as np
from numpy import linalg as LA
from scipy.io import loadmat
from scipy.signal import argrelextrema 
from numpy import fft
import time
#import pandas as pd
import timeit
from numba import jit,njit,prange,get_num_threads,set_num_threads
from attrdict import AttrDict as AttrDictSens

from .IF_aux import Maxmins, FKmask, fftconvolve2D, find_max_frequency2D, \
    get_mask_length2D, get_mask_2D_v3, compute_imf2d_fft, _compute_imf2d_fft_adv

__version__='3e'

#WRAPPER (version unaware. To be called by IF.py) 
# def IF_run(*args,**kwargs):
#     return IF_v8_3e(*args,**kwargs)

#WRAPPER (version unaware. To be called by FIF.py) 
   

################################################################################
###################### Iterative Filtering main functions ######################
################################################################################

def Settings(**kwargs):
    """
    Sets the default options to be passed to MIF
    WARNING: NO CHECK IS DONE ON THE VALIDITY OF THE INPUT
    
    WRAPPED FROM: Settings_v3.m
    NOTE:
    selecting options['imf_method'] = fft should be equivalent to do FIF
    """

    options = {}
    # General 
    #options['saveEnd'] = 0
    #options['saveInter'] = 0
    options['verbose'] = False    
    options['timeit'] = False    
    #options['plots'] = 0.0
    #options['saveplots'] = 0     
        
    # FIF
    options['delta'] = 0.001
    options['ExtPoints']=3
    options['NIMFs']=20
    options['Xi']=1.6
    options['alpha']='ave'
    options['MaxInner']=200
    options['MonotoneMaskLength']=True
    options['IsotropicMask']=True
    options['NumSteps']= 4
    options['BCmode'] = 'wrap' #'clip' #wrap
    options['Maxmins_method'] = 'zerocrossing'
    options['Maxmins_samples'] = 4
    options['imf_method'] = 'fft' #numba
    options['Extend'] = True #If true, extends the signal from nx,ny to 3nx,3ny in case a mask
                             #bigger than nx (ny) is found
    for i in kwargs:
        if i in options.keys() : options[i] = kwargs[i] 
    return AttrDictSens(options)



def MIF_run(x, options=None, M = np.array([]),**kwargs):
    
    if options is None:
        options = Settings()
    
    return MIF_v3e(x,options,M=M,**kwargs)

def FIF_run(*args,**kwargs): return MIF_run(*args,**kwargs)

def MIF_v3e(in_f,options,M=np.array([]), window_mask = None, data_mask = None, nthreads = None):

    """
    Multidimensional Iterative Filtering python implementation 
    
    
    INPUT: f: array-like, shape(M,N)
        
        THE REST OF THE CRAZY INPUT IS AS USUAL :P

    N.B. nthreads to be implemented in case of numba decomposition (also to be implemented)

    """
    f = np.copy(in_f)
    opts = AttrDictSens(options)
    #if nthreads is not None:
    #    if opts.imf_method == 'numba': 
    #        set_num_threads(nthreads)
    if opts.verbose:
        print('running IF decomposition...')
        #if verbose:
        print('****IF settings****')
        [print(i+':',options[i]) for i in options]
        print('data_mask   : ', data_mask is not None )
        if opts.imf_method == 'numba':
            print('Using nthreads: ',get_num_threads())
    tol = 1e-12 

    if opts.imf_method == 'fft': 
        compute_imf2D = compute_imf2d_fft
        #compute_imf2D = _compute_imf2d_fft_adv
    else:
        print('Only fft method implemented. Overriding...')
        compute_imf2D = compute_imf2d_fft
    #elif opts.imf_method == 'numba': 
    #    compute_imf2D = compute_imf2d_numba

    #loading master filter
    ift = opts.timeit
    if ift: 
        from . import time
        ttime = time.timeit()
        time_imfs = 0.
        time_max_nu = 0.
        time_mask = 0.
        ttime.tic

    MM = FKmask if window_mask is None else window_mask
    
    f = np.asarray(f)
    if len(f.shape) != 2: 
        raise Exception('Wrong dataset, the signal must be a 2D array!')
    
    #setting up machinery
    nx,ny = f.shape
    IMF = np.zeros([opts.NIMFs, nx,ny])
    #normalizing signal such as the maximum is +-1
    Norm1f = np.max(np.abs(f))
    f = f/Norm1f

    #NOW starting the calculation of the IMFs

    countIMFs = 0
    #Find max-frequency contained in signal
    if ift: ttime.tic 
    
    if 'M' not in locals() or np.size(M) == 0:
        N_pp, k_pp, diffMaxmins_x, diffMaxmins_y = \
            find_max_frequency2D(f,opts['Maxmins_samples'],tol=tol,\
            mode = opts.BCmode, method = opts.Maxmins_method)
    else:
        k_pp = [opts.ExtPoints*2] #[int(np.max([nx,ny])/M[countIMFs])]
        #k_pp = [int(np.max([nx,ny])/M[0])]
    
    if ift: time_max_nu += ttime.get_toc

    stats_list = np.recarray(opts.NIMFs,dtype=[('logMx',int),('logMy',int),('inStepN',int)])
    stats_list.logM = 0
    stats_list.inStepN = 0
    logM = (1,1) 
    ### Begin Iterating ###
    while countIMFs < opts.NIMFs and all([ii >= opts.ExtPoints for ii in k_pp]):
        countIMFs += 1
        print('IMF', countIMFs)
        
        h = np.copy(f)
        if 'M' not in locals() or np.size(M)<countIMFs:
            m = get_mask_length2D(opts,N_pp,k_pp,diffMaxmins_x,diffMaxmins_y,logM,countIMFs)
            if opts.verbose:
                print('\n IMF # %1.0d   -   # Extreme points (%s\n' %(countIMFs,k_pp))
        else:
            m = M[countIMFs-1]
            if type(m) is not tuple : m = (m,)*2 
            if opts.verbose:
                print('\n IMF # %1.0d   -   # mask length (%s\n' %(countIMFs,m))

        if opts.verbose:
            print('\n  step #            SD             Mask length \n\n')

        #stats = {'logM': [], 'inStepN': []}
        #stats['logM'].append(int(m))
        stats_list[countIMFs-1].logMx = int(m[0])
        stats_list[countIMFs-1].logMy = int(m[1])
        logM = (int(np.min(m)),)*2 if opts.IsotropicMask else m 
        
        if ift: ttime.tic 
        a = get_mask_2D_v3(MM, m)
        if ift: time_mask += ttime.get_toc
        #if the mask is bigger than the signal length, the decomposition ends.
        if any([ii< jj for ii,jj in zip(f.shape,a.shape)]) and not opts.Extend: 
            if opts.verbose: print('Mask length exceeds signal length. Finishing...')
            countIMFs -= 1
            break
        elif any([3*ii< jj for ii,jj in zip(f.shape,a.shape)]):
            if opts.verbose: print('Mask length exceeds three times signal length. Finishing...')
            countIMFs -= 1
            break
        
        if ift: ttime.tic 
        h, inStepN, SD = compute_imf2D(h,a,opts)
        if ift: time_imfs += ttime.get_toc
        
        if inStepN >= opts.MaxInner:
            print('Max # of inner steps reached')

        #stats['inStepN'].append(inStepN)
        stats_list[countIMFs-1].inStepN = inStepN
        
        IMF[countIMFs-1] = h
        
        f = f - h
    
        #Find max-frequency contained in residual signal
        
        if ift: ttime.tic 
        if 'M' not in locals() or np.size(M)<countIMFs+1:
            N_pp, k_pp, diffMaxmins_x, diffMaxmins_y = \
                find_max_frequency2D(f,opts['Maxmins_samples'],tol=tol, \
                mode = opts.BCmode, method = opts.Maxmins_method)
        else:
            k_pp = [opts.ExtPoints*2] #[int(np.max([nx,ny])/M[countIMFs])]
        if ift: time_max_nu += ttime.get_toc
        

    IMF = IMF[0:countIMFs]
    IMF = np.vstack([IMF, f.reshape((1,nx,ny))])
    stats_list = stats_list[:countIMFs]

    IMF = IMF*Norm1f # We scale back to the original values

    if ift: 
        ttime.total_elapsed(from_1st_start = True, hhmmss = True)
        print('imfs calculation took: %f (%.2f)' % (time_imfs,100* time_imfs / ttime._dttot)) 
        print('mask calculation took: %f (%.2f)' % (time_mask,100* time_mask / ttime._dttot)) 
        print('mask length calculation took: %f (%.2f)' % (time_max_nu,100* time_max_nu / ttime._dttot)) 

    return IMF, stats_list

