"""
 Iterative Filtering python package: general auxiliary functions

 Dependencies : numpy, scipy, numba  

 Authors: 
    Python version: Emanuele Papini - INAF (emanuele.papini@inaf.it) 
    Original matlab version: Antonio Cicone - university of L'Aquila (antonio.cicone@univaq.it)

"""

import numpy as np
from numpy import linalg as LA
from scipy.signal import argrelextrema 
from numba import njit,prange,get_num_threads,set_num_threads


from .prefixed_double_filter import MM as FKmask
FKmask = np.array(FKmask)

from attrdict import AttrDict
AttrDictSens = AttrDict


@njit
def lanorm(x,ordd):
    return LA.norm(x,ordd)

def Maxmins(x, tol = 1e-12, mode = 'clip', method = 'zerocrossing'):
    """
    Finds the indices of the extrema (maxima and minima) contained in x, or 
    in its derivative (if 'zerocrossing' is selected).
    
    Parameters
    ----------

    x : 1D-array like
        array of the signal to find the extrema of
    tol : float
        used if 'zerocrossing' method is selected (see below)
    mode : str
        type of boundary conditions (see scipy.signal.argrelextrema)
        only works with method 'argrelextrema'
    method: str
        'argrelextrema': compute maxima and minima using argrelextrema. Ignores tol
        'zerocrossing' : compute maxima and minima using zero crossing of 1st derivative.
                         If diff through the crossing is less than tol, the point is ignored. 
    """
    if method == 'argrelextrema':
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
    
    parameters
    ----------
    f : 1D array-like
        input array of which to find the max frequency

    **kwargs: optional parameters to be passed to Maxmins

    returns
    -------
    N_pp : int, f.size
    k_pp : int, number of extrema found
    maxmins: array of integer (size k_pp), index (location) of extrema
    diffMaxmins: array of integer (size k_pp -1): distance between neighbohr extrema
    
    """
    
    maxmins = Maxmins(f,**kwargs)
    if np.size(maxmins) < 2:
        print('No extrema detected')
        #return (N,M),[0,0],[],[]
        return np.size(f),0,None,None
    
    diffMaxmins = np.diff(maxmins)
    
    N_pp = np.size(f)
    k_pp = maxmins.shape[0]

    return N_pp, k_pp, maxmins, diffMaxmins


def get_mask_length(options,N_pp,k_pp,diffMaxmins_pp,logM,countIMFs):
    """
    Compute the mask length (in points) given a certain set of input parameters
    
    The input parameters are the one given by find_max_frequency
    
    parameters
    ----------
    options : dict
        dict containing at least the following mandatory keys
        'alpha' : str or float
            'ave': the mask length is computed using the average frequency
                i.e. 2*N_pp/k_pp*options.Xi
            'Almost_min' : compute the mask length using the minimum between
                the 'ave' formula and 30th-percentile diffMaxmins_pp
            'Median' : compute the mask length using the median of the distribution of 
                diffMaxmins_pp multiplied by the stretching factor options.Xi
            float: compute the mask length using the percentile specified in options['alpha']
                then multiplied by the stretching factor options['Xi']
        'Xi' : float
            stretching factor for computing the mask length

    N_pp : int
        length of the signal in term of points. (aka, its size)
    k_pp : int
        number of extrema present in the signal 
    diffMaxmins: array of integer (size k_pp -1) 
        distance between neighbohr extrema
    logM : int
        minimum mask-length allowed. If the mask length is found to be less than logM,
        then override the value of the new length to logM*1.1
    countIMF : int
        number of the ith IMF (IMC) that we are actually computing
        if countIMF <= 1, the logM argument is not used, and the computed mask is returned.
    """

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
    k could be either an integer or a float.
    y is the area under the curve for each bar

    Parameters
    ----------
    y : 1D-array like
        input mask function
    k : int or float
        mask length
    verbose : bool
        set verbose level
    tol : float
        tolerance to be used in the normalization check
    
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
    """
    Extracts the imf from the signal f using the window function (mask) a,
    according to the settings specified in the options dict
    
    N.B. This calculation is done via convolution of f with a in normal space,
    using numba to accelerate the computations.
    
    Parameters
    ----------
    f : 1D float array
        input signal
    a : 1D float array
        window function
    options : dict
        dictionary containing the settings of the decomposition 
        (see, e.g.,  fifpy.IFpy.Settings method).
    """
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
    """
    Extracts the imf from the signal f using the window function (mask) a,
    according to the settings specified in the options dict
    
    N.B. This calculation is done via convolution of f with a in Fourier space,
    using scipy.signal.fftconvolve
    
    Parameters
    ----------
    f : 1D float array
        input signal
    a : 1D float array
        window function
    
    options : dict
        dictionary containing the settings of the decomposition 
        (see, e.g.,  fifpy.IFpy.Settings method).
        mandatory keyword in options:
        'delta' : minimum difference in the 2norm between the two iterations
        'MaxInner': maximum number of iterations
        'verbose' : verbosity level 
    """
        
    from scipy.signal import fftconvolve

    h = np.array(f)
    h_ave = np.zeros(len(h))

    kernel = a
    delta = options.delta
    MaxInner = options.MaxInner
    BCmod = options.BCmode 
        
    inStepN = 0
    SD = 1.
    
    Nh = len(h)
    while SD>delta and inStepN<MaxInner:
        inStepN += 1
        if BCmod == 'wrap':
            h_ave = fftconvolve1D(h,kernel)
        else:
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
    WARNING: IT DOESN'T WORK FOR SCIPY NEW VERSIONS (python>3.10.xx) 
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

def fftconvolve1D(f,ker):#, mode = 'same', BCmode = 'wrap'):
    """
    
    Compute the 1D convolution between f and ker, using fft.
    
    It assumes that the field is periodic 

    This function is used when the option "extend-periodic" is selected

    
    parameters
    ----------
    f : 1D-like array
        input array
    ker : 1D-like array
        kernel of the convolution filter

    """
    if f.shape[0] <ker.shape[0]:
        print('error, kernel shape cannot be larger than 1D array shape')
        return None
    
    m = ker.shape[0]//2
    kpad = np.pad(ker,((0,f.shape[0]-ker.shape[0])))
    kpad = np.roll(kpad,-m)
    return np.fft.irfft(np.fft.rfft(f)*np.fft.rfft(kpad),n=f.shape[0])

################################################################################
#################### MIF SPECIFIC AUXILIARY FUNCTIONS ##########################
################################################################################

def fftconvolve2D(f,ker):#, mode = 'same', BCmode = 'wrap'):
    """
    
    Compute the 2D convolution between f and ker, using fft.
    
    It assumes that the field is biperiodic

    SO FAR only mode = 'same' and BCmode = 'wrap' are implemented
    maybe use pylab_convolve2D for direct convolution
    
    parameters
    ----------
    f : 2D-like array
        input array
    ker : 2D-like array
        kernel of the convolution filter

    """
    if any([i<j for i,j in zip(f.shape,ker.shape)]):
        print('error, kernel shape cannot be larger than 2D array shape')
        return None
    m = [i//2 for i in ker.shape]
    kpad = np.pad(ker,((0,f.shape[0]-ker.shape[0]),(0,f.shape[1]-ker.shape[1])))
    kpad = np.roll(kpad,(-m[0],-m[1]),(0,1))
    return np.fft.irfft2(np.fft.rfft2(f)*np.fft.rfft2(kpad),s=f.shape) 

def find_max_frequency2D(f,nsamples = 1, **kwargs):
    """
    find extrema contained in f and returns
    N_pp,k_pp,maxmins,diffMaxmins

    N_pp : int, f.size
    k_pp : int, number of extrema found
    maxmins: array of integer (size k_pp), index (location) of extrema
    diffMaxmins: array of integer (size k_pp -1): distance between neighbohr extrema
    
    Parameters
    ----------
    f : 2D array
        input 2D signal
    nsamples : int
        number of cut along which to calculate the ma frequency
    kwargs : dict
        optional kwargs to be passed to Maxmins
    """
    from random import randrange
    
    N,M = f.shape
    kn = N//nsamples
    km = M//nsamples
    k_pp = []
    diffMaxmins_x = []
    diffMaxmins_y = []
    for i in range(nsamples):
        
        #this is to avoid that None are produced
        for ix in range(kn):
            maxmins_x = Maxmins(f[randrange(N),:].flatten(),**kwargs)
            if maxmins_x is not None: break
        
        for iy in range(km):
            maxmins_y = Maxmins(f[:,randrange(M)].flatten(),**kwargs)
            if maxmins_y is not None: break
        
        if maxmins_x is not None and maxmins_y is not None:
            diff_x = [] if np.size(maxmins_x) < 2 else np.diff(maxmins_x)
            diff_y = [] if np.size(maxmins_y) < 2 else np.diff(maxmins_y)
            
            k_pp.append([np.size(maxmins_x),np.size(maxmins_y)]) 
            diffMaxmins_x.append(diff_x) 
            diffMaxmins_y.append(diff_y) 
    if np.sum(k_pp) == 0.:
        print('No extrema detected')
        return (N,M),[0,0],[],[]

    #getting maximum number of extrema detected along x and y
    k_pp = np.array(k_pp).squeeze().max(axis=0)

    #flattening distribution of maxmins distances to be used for percentile
    #calculation (see get_mask_length)
    diffMaxmins_x = np.concatenate([i.flatten() for i in diffMaxmins_x if np.size(i) > 0])
    diffMaxmins_y = np.concatenate([i.flatten() for i in diffMaxmins_y if np.size(i) > 0])
    N_pp = (N,M)

    return N_pp, k_pp, diffMaxmins_x,diffMaxmins_y


def get_mask_length2D(options,N_pp,k_pp,diffMaxmins_x,diffMaxmins_y,logM,countIMFs):
    """
    calculate mask length from a 2D signal 

    Parameters:
    options : dict
        dictionary of Settings to be used in the calculation (see, e.g., fifpy.MIF.Settings)
    
    N_pp, k_pp, diffMAxmins_x,diffMaxmins_y are given as output of find_max_frequency2D

    logM : list of 2 ints
        mask length in 2D from the previous IMF, used to check whether the new calculated
        mask is smaller than that.
    countIMFs : int
        if > 1, then a check on logM is performed. If 1, then no check is done since it is the
        first IMC.
    """ 
    if isinstance(options.alpha,str):
    
        if options.alpha == 'ave': 
            mx = 2*np.round(N_pp[0]/k_pp[0]*options.Xi)
            my = 2*np.round(N_pp[1]/k_pp[1]*options.Xi)
        elif options.alpha == 'Almost_min': 
            mx = 2*np.min( [options.Xi*np.percentile(diffMaxmins_x,30), np.round(N_pp[0]/k_pp[0]*options.Xi)])
            my = 2*np.min( [options.Xi*np.percentile(diffMaxmins_y,30), np.round(N_pp[1]/k_pp[1]*options.Xi)])
        elif options.alpha == 'Median':
            mx = 2*np.round(np.median(diffMaxmins_x)*options.Xi)
            my = 2*np.round(np.median(diffMaxmins_y)*options.Xi)
        else:    
            raise Exception('Value of alpha not recognized!\n')
    
    else:
        mx = 2*np.round(options.Xi*np.percentile(diffMaxmins_x,options.alpha))
        my = 2*np.round(options.Xi*np.percentile(diffMaxmins_y,options.alpha))
    
    if countIMFs > 1:
        if mx <= logM[0]:
            if options.verbose:
                print('Warning mask length (x) is decreasing at step %1d. ' % countIMFs)
            if options.MonotoneMaskLength:
                mx = np.ceil(logM[0] * 1.1)
                if options.verbose:
                    print('The old mask length (x) is %1d whereas the new one is forced to be %1d.\n' % (
                    logM[0], mx))
            else:
                if options.verbose:
                    print('The old mask length (x) is %1d whereas the new one is %1d.\n' % (logM[0], mx))
        if my <= logM[1]:
            if options.verbose:
                print('Warning mask length (y) is decreasing at step %1d. ' % countIMFs)
            if options.MonotoneMaskLength:
                my = np.ceil(logM[1] * 1.1)
                if options.verbose:
                    print('The old mask length (y) is %1d whereas the new one is forced to be %1d.\n' % (
                    logM[1], my))
            else:
                if options.verbose:
                    print('The old mask length (y) is %1d whereas the new one is %1d.\n' % (logM[1], my))

    return int(mx),int(my)

def get_mask_2D_v3(w,k):
    """
    get the mask with length 2*k+1 x 2*k+1
    
    Parameters
    ----------
    k : integer, half mask length
    w : is the area under the curve for each bar
    output: a 2D mask with shape (2*k+1, 2*k+1)
    wrapped from FIF2_v3.m
    """
    #check if k tuple contains integers
    if not all([type(i) is int for i in k]):
        print('input mask not integer, making it so')
        k=tuple([int(i) for i in k])

    L=np.size(w)
    m=(L-1)/2;  
    w = np.pad(w,(0,(L-1)//2))
    A=np.zeros((2*k[0]+1,2*k[1]+1))
    if all([i<=m for i in k]): # The prefixed filter contains enough points
        #distance matrix
        xx = np.arange(-k[0],k[0]+1)/k[0]
        yy = np.arange(-k[1],k[1]+1)/k[1]
        dm = np.sqrt(xx[:,None]**2 + yy[None,:]**2) #normalized distance from ellipse border
        s = (m-1)+L/2*dm
        t = s+2
        s2 = np.ceil(s) - s
        t1 = t - np.floor(t)
        for i in range(2*k[0]+1):
            for j in range(2*k[1]+1):
                A[i,j] = np.sum(w[int(np.ceil(s[i,j]))-1:int(t[i,j])]) +\
                         s2[i,j] * w[int(np.ceil(s[i,j]))-1] + t1[i,j]* w[int(t[i,j])-1]
        A/=np.sum(A)
    else : #We need a filter with more points than MM, we use interpolation
        print('Need to write the code!')
        A=[]

    return A

def compute_imf2d_fft(f,a,options):
    """
    Calculates the IMC from the  2D signal f using the 2D window function a.

    Parameters
    ----------
    f : 2D array
        input signal from which to extract the IMC
    a : 2D array
        input 2D window function
    options : dict
        dictionary of Settings to be used in the decomposition
        (see fifpy.MIFpy.Settings)
    """ 

    h = np.array(f)
    h_ave = np.zeros(len(h))
 
    kernel = a
    delta = options.delta
    MaxInner = options.MaxInner
    
    inStepN = 0
    SD = 1.
    #checking whether to extend the signal
    if options.Extend and any([i<j for i,j in zip(h.shape,kernel.shape)]):
        h = np.pad(h,(h.shape,)*2,mode='wrap')
        print(h.shape)
    Nh = len(h)
    while SD>delta and inStepN<MaxInner:
        inStepN += 1
        h_ave = fftconvolve2D(h,kernel)#,mode='same', BCmode = 'wrap')
        #computing norm
        SD = LA.norm(h_ave)**2/LA.norm(h)**2
        h = h[...] - h_ave[...]
        h_ave[...] = 0
        if options.verbose:
            print('(fft): %2.0d      %1.40f          %s\n' % (inStepN, SD, np.shape(a)))
    
    if options.verbose:
        print('(fft): %2.0d      %1.40f          %s\n' % (inStepN, SD, np.shape(a)))
    if h.shape !=f.shape:
        nx,ny = f.shape
        h = h[nx:2*nx,ny:2*ny]

    return h,inStepN,SD
def _compute_imf2d_fft_adv(f,a,options):
        
    h = np.array(f)
    h_ave = np.zeros(len(h))
    
    
    ker = a
    delta = options.delta
    MaxInner = options.MaxInner
    ksteps =options.NumSteps
    
    inStepN = 0
    SD = 1.
    
    Nh = len(h)
    if options.Extend and any([i<j for i,j in zip(h.shape,ker.shape)]):
        h = np.pad(h,(h.shape,)*2,mode='wrap')
    
    if any([i<j for i,j in zip(h.shape,ker.shape)]):
        print('error, kernel shape cannot be larger than 2D array shape')
        return None,None,None
    
    m = [i//2 for i in ker.shape]
    kpad = np.pad(ker,((0,h.shape[0]-ker.shape[0]),(0,h.shape[1]-ker.shape[1])))
    kpad = np.roll(kpad,(-m[0],-m[1]),(0,1))
    fkpad = np.fft.rfft2(kpad); del kpad
    fh = np.fft.rfft2(h)
    while SD>delta and inStepN<MaxInner:
        inStepN += ksteps
        fh_ave = (1-fkpad)**inStepN * fh
        fh_avem1 = (1-fkpad)**(inStepN-1) * fh
        SD = np.abs((np.abs(fh_ave)**2).sum()/(np.abs(fh_avem1)**2).sum() -1)
    
        if options.verbose:
            print('(fft_adv): %2.0d      %1.40f          %s\n' % (inStepN, SD, np.shape(a)))
    
    if options.verbose:
        print('(fft_adv): %2.0d      %1.40f          %s\n' % (inStepN, SD, np.shape(a)))
    
    h = np.fft.irfft2(fh_ave,s=h.shape) 
    if h.shape !=f.shape:
        nx,ny = f.shape
        h = h[nx:2*nx,ny:2*ny]
    return h,inStepN,SD
