"""
 Iterative Filtering python package: general auxiliary functions

 Dependencies : numpy, scipy, numba, attrdict  //TO CHECK

 Authors: 
    Python version: Emanuele Papini - INAF (emanuele.papini@inaf.it) 
    Original matlab version: Antonio Cicone - university of L'Aquila (antonio.cicone@univaq.it)

"""

import os
import numpy as np
from numpy import linalg as LA
from scipy.io import loadmat
from scipy.signal import argrelextrema 
from numpy import fft
from numba import jit,njit,prange,get_num_threads,set_num_threads

#THIS is a full substitute for my custom class that doesn't perform very well
from attrdict import AttrDict as AttrDictSens


def get_window_file_path():
    import sys
    _path_=sys.modules[__name__].__file__.split(sep='/')
    pp=_path_[0]
    for ipath in _path_[1:-1]:
        pp+='/'+ipath
    return pp+'/prefixed_double_filter.mat'

@njit
def lanorm(x,ordd):
    return LA.norm(x,ordd)

def Maxmins(x, tol = 1e-12, mode = 'clip', method = 'zerocrossing'):
    """
    method: str
        'argrelextrema': compute maxima and minima using argrelextrema. Ignores tol
        'zerocrossing' : compute maxima and minima using zero crossing of 1st derivative.
                         If diff through the crossing is less than tol, the point is ignored. 
    """
    if method == 'argrelextrema':
        from scipy.signal import argrelextrema
        maxima = argrelextrema(x, np.greater, mode = mode)
        minima = argrelextrema(x, np.less, mode = mode)

        extrema = np.sort(np.concatenate((maxima, minima), axis=1))
    elif method == 'zerocrossing':
        dx = np.diff(x) #naive derivative
        sgns = np.diff(np.sign(dx)) #location of maxmins: max/min where sgns = -2/+2
        extrema = np.where(sgns != 0)[0]
        if len(extrema) < 1: return None
        dx = np.abs(dx[1:][extrema])
        extrema = extrema[dx>tol]
        if len(extrema) < 1: return None
        extrema +=1
    else: raise Exception('WRONG INPUT METHOD!')

    return extrema.squeeze()
   

def find_max_frequency(f, **kwargs):
    """
    find extrema contained in f and returns
    N_pp,k_pp,maxmins,diffMaxmins

    N_pp : int, f.size
    k_pp : int, number of extrema found
    maxmins: array of integer (size k_pp), index (location) of extrema
    diffMaxmins: array of integer (size k_pp -1): distance between neighbohr extrema
    """
    
    maxmins = Maxmins(f,**kwargs)
    if len(maxmins) < 1:
        print('No extrema detected')
        return None,None
    
    diffMaxmins = np.diff(maxmins)
    
    N_pp = len(f)
    k_pp = maxmins.shape[0]

    return N_pp, k_pp, maxmins, diffMaxmins


def get_mask_length(options,N_pp,k_pp,diffMaxmins_pp,logM,countIMFs):
    
    if isinstance(options.alpha,str):
    
        if options.alpha == 'ave': 
            m = 2*np.round(N_pp/k_pp*options.Xi)
        elif options.alpha == 'Almost_min': 
            m = 2*np.min( [options.Xi*np.percentile(diffMaxmins_pp,30), np.round(N_pp/k_pp*options.Xi)])
        elif options.alpha == 'Median':
            m = 2*np.round(np.median(diffMaxmins_pp)*options.Xi)
        else:    
            raise Exception('Value of alpha not recognized!\n')
    
    else:
        m = 2*np.round(options.Xi*np.percentile(diffMaxmins_pp,options.alpha))
    
    if countIMFs > 1:
        if m <= logM:
            if options.verbose:
                print('Warning mask length is decreasing at step %1d. ' % countIMFs)
            if options.MonotoneMaskLength:
                m = np.ceil(logM * 1.1)
                if options.verbose:
                    print('The old mask length is %1d whereas the new one is forced to be %1d.\n' % (
                    logM, m))
            else:
                if options.verbose:
                    print('The old mask length is %1d whereas the new one is %1d.\n' % (logM, m))

    return m

def get_mask_v1_1(y, k,verbose,tol):
    """
    Rescale the mask y so that its length becomes 2*k+1.
    k could be an integer or not an integer.
    y is the area under the curve for each bar
    
    wrapped from FIF_v2_13.m
    
    """
    n = np.size(y)
    m = (n-1)//2
    k = int(k)

    if k<=m:

        if np.mod(k,1) == 0:
            
            a = np.zeros(2*k+1)
            
            for i in range(1, 2*k+2):
                s = (i-1)*(2*m+1)/(2*k+1)+1
                t = i*(2*m+1)/(2*k+1)

                s2 = np.ceil(s) - s

                t1 = t - np.floor(t)

                if np.floor(t)<1:
                    print('Ops')

                a[i-1] = np.sum(y[int(np.ceil(s))-1:int(np.floor(t))]) +\
                         s2*y[int(np.ceil(s))-1] + t1*y[int(np.floor(t))-1]
        else:
            new_k = int(np.floor(k))
            extra = k - new_k
            c = (2*m+1)/(2*new_k+1+2*extra)

            a = np.zeros(2*new_k+3)

            t = extra*c + 1
            t1 = t - np.floor(t)

            if k<0:
                print('Ops')
                a = []
                return a

            a[0] = np.sum(y[:int(np.floor(t))]) + t1*y[int(np.floor(t))-1]

            for i in range(2, 2*new_k+3):
                s = extra*c + (i-2)*c+1
                t = extra*c + (i-1)*c
                s2 = np.ceil(s) - s
                t1 = t - np.floor(t)

                a[i-1] = np.sum(y[int(np.ceil(s))-1:int(np.floor(t))]) +\
                         s2*y[int(np.ceil(s))-1] + t1*y[int(np.floor(t))-1]
            t2 = np.ceil(t) - t

            a[2*new_k+2] = np.sum(y[int(np.ceil(t))-1:n]) + t2*y[int(np.ceil(t))-1]

    else: # We need a filter with more points than MM, we use interpolation
        dx = 0.01
        # we assume that MM has a dx = 0.01, if m = 6200 it correspond to a
        # filter of length 62*2 in the physical space
        f = y/dx
        dy = m*dx/k
        # b = np.interp(list(range(1,int(m+1),m/k)), list(range(0,int(m+1))), f[m:2*m+1])
        b = np.interp(np.linspace(0,m,int(np.ceil(k+1))), np.linspace(0,m,m+1), f[m:2*m+1])

        a = np.concatenate((np.flipud(b[1:]), b))*dy

        if abs(LA.norm(a,1)-1)>tol:
            if verbose:
                print('\n\n Warning!\n\n')
                print(' Area under the mask equals %2.20f\n'%(LA.norm(a,1),))
                print(' it should be equal to 1\n We rescale it using its norm 1\n\n')
            a = a/LA.norm(a,1)
        
    return a

################################################################################
##################### MAIN IMF CALCULATION FUNCTIONS ###########################
################################################################################

def compute_imf_numba(f,a,options):
        
    h = np.array(f)
    h_ave = np.zeros(len(h))

    @njit(parallel=True)
    def iterate_numba(h,h_ave,kernel,delta,MaxInner):
        
        inStepN = 0
        SD = 1.
        
        ker_size = len(kernel)
        hker_size = ker_size//2
        #kernel[hker_size] -=1 #so we get the high frequency filter 
        
        Nh = len(h)
        while SD>delta and inStepN<MaxInner:
            inStepN += 1
            #convolving the function with the mask: High Pass filter
            #convolving edges (influence cone)
            for i in prange(hker_size):
                for j in range(i+hker_size+1): 
                    h_ave[i] += h[j] * kernel[hker_size-i+j] 
                    h_ave[-i-1] += h[-j-1] * kernel[hker_size+i-j] 
            #convolving inner part
            for i in prange(hker_size, Nh - hker_size):
                for j in range(ker_size): h_ave[i] += h[i-hker_size+j] * kernel[j] 

            #computing norm
            SD =  0
            hnorm = 0
            for i in range(Nh):
                SD+= h_ave[i]**2
                hnorm+= h[i]**2
            SD /=hnorm
            h[:] = h[:] - h_ave[:]
            h_ave[:] = 0
        
        return h,inStepN,SD

    h_ave, inStepN, SD = iterate_numba(h,h_ave,a,options.delta,options.MaxInner)
    
    if options.verbose:
        print('(numba): %2.0d      %1.40f          %2.0d\n' % (inStepN, SD, np.size(a)))

    return h_ave,inStepN,SD

def compute_imf_fft(f,a,options):
        
    from scipy.signal import fftconvolve

    h = np.array(f)
    h_ave = np.zeros(len(h))

    kernel = a
    delta = options.delta
    MaxInner = options.MaxInner
    
        
    inStepN = 0
    SD = 1.
    
    Nh = len(h)
    while SD>delta and inStepN<MaxInner:
        inStepN += 1
        h_ave = fftconvolve(h,kernel,mode='same')
        #computing norm
        SD = LA.norm(h_ave)**2/LA.norm(h)**2
        h[:] = h[:] - h_ave[:]
        h_ave[:] = 0
        if options.verbose:
            print('(fft): %2.0d      %1.40f          %2.0d\n' % (inStepN, SD, np.size(a)))
    

    
    if options.verbose:
        print('(fft): %2.0d      %1.40f          %2.0d\n' % (inStepN, SD, np.size(a)))

    return h,inStepN,SD

def compute_imf_fft_adv(f,a,options):
    """
    slightly faster since it spares the calculation of 1fft per iteration
    """
    from scipy.signal.signaltools import _init_freq_conv_axes, _centered
    from scipy import fft as sp_fft
    
    h = np.array(f)
    h_ave = np.zeros(len(h))

    kernel = a
    delta = options.delta
    MaxInner = options.MaxInner
    
        
    inStepN = 0
    SD = 1.
    
    Nh = len(h)
    #setting machinery for fftconvolve directly wrapped from scipy.signal.signaltools
    in1 = h; in2 = kernel
    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])
    in1, in2, axes = _init_freq_conv_axes(in1, in2, 'same', None,
                                          sorted_axes=False)
    s1 = in1.shape
    s2 = in2.shape

    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
             for i in range(in1.ndim)]
    
    calc_fast_len=True
    if not len(axes):
        return in1 * in2

    complex_result = (in1.dtype.kind == 'c' or in2.dtype.kind == 'c')

    if calc_fast_len:
        # Speed up FFT by padding to optimal size.
        fshape = [
            sp_fft.next_fast_len(shape[a], not complex_result) for a in axes]
    else:
        fshape = shape

    if not complex_result:
        fft, ifft = sp_fft.rfftn, sp_fft.irfftn
    else:
        fft, ifft = sp_fft.fftn, sp_fft.ifftn

    #sp1 = fft(in1, fshape, axes=axes)
    sp2 = fft(in2, fshape, axes=axes)

    
    while SD>delta and inStepN<MaxInner:
        inStepN += 1
        sp1 = fft(in1, fshape, axes=axes)
        ret = ifft(sp1 * sp2, fshape, axes=axes)
        if calc_fast_len:
            fslice = tuple([slice(sz) for sz in shape])
            ret = ret[fslice]
    
        h_ave =  _centered(ret, s1).copy() 

        #computing norm
        SD = LA.norm(h_ave)**2/LA.norm(h)**2
        h[:] = h[:] - h_ave[:]
        h_ave[:] = 0
        if options.verbose:
            print('(fft adv): %2.0d      %1.40f          %2.0d\n' % (inStepN, SD, np.size(a)))
    

    
    if options.verbose:
        print('(fft adv): %2.0d      %1.40f          %2.0d\n' % (inStepN, SD, np.size(a)))

    return h,inStepN,SD
