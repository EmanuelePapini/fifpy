"""
 Iterative Filtering python package

 Dependencies : numpy, scipy, numba, joblib  //TO CHECK

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
import time
#import pandas as pd
import timeit
from numba import jit,njit,prange,get_num_threads,set_num_threads

__version__='8.3e'

#WRAPPER (version unaware. To be called by IF.py) 
# def IF_run(*args,**kwargs):
#     return IF_v8_3e(*args,**kwargs)

#WRAPPER (version unaware. To be called by FIF.py) 

################################################################################
########################## AUXILIARY FUNCTIONS #################################
################################################################################
class AttrDictSens(dict):
    '''
    A case-sensitive dictionary with access via item, attribute, and call
    notations:

        >>> d = AttrDict()
        >>> d['Variable'] = 123
        >>> d['Variable']
        123
        >>> d.Variable
        123
        >>> d.variable
        123
        >>> d('VARIABLE')
        123
    '''

    def __init__(self, init={}):
        dict.__init__(self, init)

    def __getitem__(self, name):
        try :
            return super(AttrDictSens, self).__getitem__(name)
        except:
            raise AttributeError

    def __setitem__(self, key, value):
        return super(AttrDictSens, self).__setitem__(key, value)

    __getattr__ = __getitem__
    __setattr__ = __setitem__
    __call__ = __getitem__

def get_window_file_path():
    import sys
    _path_=sys.modules[__name__].__file__[0:-12]
    return _path_+'/prefixed_double_filter.mat'


################################################################################
###################### Iterative Filtering aux functions #######################
################################################################################
#def Maxmins(x,tol,mode='open'):
#    """
#    Wrapped from: Maxmins_v3_8 in FIF_v2_13.m
#    """
#    @jit(nopython=True) 
#    def maxmins_wrap(x,df, N,Maxs,Mins):
#
#        h = 1
#        while h<N and np.abs(df[h]/x[h]) <= tol:
#            h = h + 1
#   
#        if h==N:
#            return None, None
#
#        cmaxs = 0
#        cmins = 0
#        c = 0
#        N_old = N
#        
#        df = np.zeros(N+h)
#        df[0:N] = x
#        df[N:N+h] = x[1:h+1]
#        for i in range(N+h-1):
#            df[i] = df[i+1] - df[i]
#        
#        f = np.zeros(N+h-1)
#        f[0:N] = x
#        f[N:] = f[1:h]
#        
#        N = N+h
#        #beginfor
#        for i in range(h-1,N-2):
#            if abs(df[i]*df[i+1]/f[i]**2) <= tol :
#                if df[i]/abs(f[i]) < -tol:
#                    last_df = -1
#                    posc = i
#                elif df[i]/abs(f[i]) > tol:
#                    last_df = +1
#                    posc = i
#                elif df[i] == 0:
#                    last_df = 0
#                    posc = i
#
#                c = c+1
#
#                if df[i+1]/abs(f[i]) < -tol:
#                    if last_df == 1 or last_df == 0:
#                        cmaxs = cmaxs +1
#                        Maxs[cmaxs] = (posc + (c-1)//2 +1)%N_old
#                    c = 0
#                
#                if df[i+1]/abs(f[i]) > tol:
#                    if last_df == -1 or last_df == 0:
#                        cmins = cmins +1
#                        Mins[cmins] = (posc + (c-1)//2 +1)%N_old
#                    c = 0
#
#            if df[i]*df[i+1]/f[i]**2 < -tol:
#                if df[i]/abs(f[i]) < -tol and df[i+1]/abs(f[i]) > tol:
#                    cmins  =cmins+1
#                    Mins[cmins] = (i+1)%N_old
#                    if Mins[cmins]==0:
#                        Mins[cmins]=1
#                    last_df=-1
#
#                elif df[i]/abs(f[i]) > tol and df[i+1]/abs(f[i])  < -tol:
#                    cmaxs = cmaxs+1
#                    Maxs[cmaxs] = (i+1)%N_old
#                    if Maxs[cmaxs] == 0:
#                        Maxs[cmaxs]=1
#            
#                    last_df =+1
#
#        if c>0:
#            if cmins>0 and Mins[cmins] == 0 : Mins[cmins] = N
#            if cmaxs>0 and Maxs[cmaxs] == 0 : Maxs[cmaxs] = N
#
#        return Maxs[0:cmaxs], Mins[0:cmins]
#
#    N = np.size(x)
#
#    Maxs = np.zeros(N)
#    Mins = np.zeros(N)
#    
#    df = np.diff(x)
#
#    if mode == 'wrap':
#        Maxs, Mins = maxmins_wrap(x,df,N,Maxs,Mins)
#        if Maxs is None or Mins is None:
#            return None,None,None
#
#        maxmins = np.sort(np.concatenate((Maxs,Mins) ))
#        
#        if any(Mins ==0): Mins[Mins == 0] = 1
#        if any(Maxs ==0): Maxs[Maxs == 0] = 1
#        if any(maxmins ==0): maxmins[maxmins == 0] = 1
#
#    return maxmins,Maxs,Mins
#def find_max_frequency(f,tol, mode = 'clip'):
#
#    f_pp = np.delete(f, np.argwhere(abs(f)<=tol))
#    if np.size(f_pp) < 1: 
#        print('Signal too small')
#        return None,None
#
#    maxmins_pp = Maxmins(np.concatenate([f_pp, f_pp[:10]]),tol,mode = mode)    
#    maxmins_pp = maxmins_pp[0] 
#    if len(maxmins_pp) < 1:
#        print('No extrema detected')
#        return None,None
#    
#    maxmins_pp = maxmins_pp[maxmins_pp<f_pp.size]
#
#    diffMaxmins_pp = np.diff(maxmins_pp)
#    
#    N_pp = len(f_pp)
#    k_pp = maxmins_pp.shape[0]

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
def _compute_imf_fft_adv(f,a,options):
    """
    next step is doing the norm calculation in Fourier space so to reduce 
    to only 2 FFTs
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
################################################################################
###################### Iterative Filtering main functions ######################
################################################################################

def Settings(**kwargs):
    """
    Sets the default options to be passed to xFIF
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
    options['NIMFs']=200
    options['Xi']=1.6
    options['alpha']='ave'
    options['MaxInner']=200
    options['MonotoneMaskLength']=True
    options['NumSteps']=1
    options['BCmode'] = 'clip' #wrap
    options['Maxmins_method'] = 'zerocrossing'
    options['imf_method'] = 'fft' #numba
    for i in kwargs:
        if i in options.keys() : options[i] = kwargs[i] 
    return AttrDictSens(options)

def IF_run(x, options=None, M = np.array([]),**kwargs):
    
    if options is None:
        options = Settings()
    
    return IF_v8_3e(x,options,M=M,**kwargs)


def IF_v8_3e(f,options,M=np.array([]), window_file=None, data_mask = None, nthreads = None):

    """
    Iterative Filtering python implementation (version 8) Parallel version
    adapted from MvFIF_v8.m
    
    INPUT: x: array-like, shape(D,N)
        
        THE REST OF THE CRAZY INPUT IS AS USUAL :P

        the FIF decomposition is along N, and the analysis is performed on
        D channels/signals at once omogenizing the window size to be the same
        i.e., it is a concurrent decomposition

    data_mask : None or boolean array of size x
        used to mask data that wont be used to determine the size of the window mask (LogM).
    """
    opts = AttrDictSens(options)
    if nthreads is not None:
        if opts.imf_method == 'numba': 
            set_num_threads(nthreads)
    if opts.verbose:
        print('running IF decomposition...')
        #if verbose:
        print('****IF settings****')
        [print(i+':',options[i]) for i in options]
        print('data_mask   : ', data_mask is not None )
        if opts.imf_method == 'numba':
            print('Using nthreads: ',get_num_threads())
    tol = 1e-18 

    if opts.imf_method == 'fft': 
        compute_imf = compute_imf_fft
        #compute_imf = _compute_imf_fft_adv
    elif opts.imf_method == 'numba': 
        compute_imf = compute_imf_numba

    #loading master filter
    ift = opts.timeit
    if ift: 
        from . import time
        ttime = time.timeit()
        time_imfs = 0.
        time_max_nu = 0.
        time_mask = 0.
        ttime.tic

    if window_file is None:
        window_file = get_window_file_path()
    try:
        MM = loadmat(window_file)['MM'].flatten()
    except:
        raise ValueError("ERROR! Could not load window function from file: "+window_file)
    f = np.asarray(f)
    if len(f.shape) > 1: 
        raise Exception('Wrong dataset, the signal must be a 1D array!')
    
    #setting up machinery
    N = f.size
    IMF = np.zeros([opts.NIMFs, N])
    #normalizing signal such as the maximum is +-1
    Norm1f = np.max(np.abs(f))
    f = f/Norm1f

    #NOW starting the calculation of the IMFs

    #Find max-frequency contained in signal
    if ift: ttime.tic 
    N_pp, k_pp, maxmins_pp, diffMaxmins_pp = find_max_frequency(f,tol=tol, mode = opts.BCmode, method = opts.Maxmins_method)
    if ift: time_max_nu += ttime.get_toc

    countIMFs = 0
    stats_list = np.recarray(opts.NIMFs,dtype=[('logM',int),('inStepN',int)])
    stats_list.logM = 0
    stats_list.inStepN = 0
    logM = 1 
    ### Begin Iterating ###
    while countIMFs < opts.NIMFs and k_pp >= opts.ExtPoints:
        countIMFs += 1
        print('IMF', countIMFs)
        
        h = f
        if 'M' not in locals() or np.size(M)<countIMFs:
            m = get_mask_length(opts,N_pp,k_pp,diffMaxmins_pp,logM,countIMFs)
        else:
            m = M[countIMFs-1]
        
        if opts.verbose:
            print('\n IMF # %1.0d   -   # Extreme points %5.0d\n' %(countIMFs,k_pp))
            print('\n  step #            SD             Mask length \n\n')

        #stats = {'logM': [], 'inStepN': []}
        #stats['logM'].append(int(m))
        stats_list[countIMFs-1].logM = int(m)
        logM = int(m)
        
        if ift: ttime.tic 
        a = get_mask_v1_1(MM, m,opts.verbose,tol)
        if ift: time_mask += ttime.get_toc
        #if the mask is bigger than the signal length, the decomposition ends.
        if N < np.size(a): 
            if opts.verbose: print('Mask length exceeds signal length. Finishing...')
            countIMFs -= 1
            break

        if ift: ttime.tic 
        h, inStepN, SD = compute_imf(h,a,opts)
        if ift: time_imfs += ttime.get_toc
        
        if inStepN >= opts.MaxInner:
            print('Max # of inner steps reached')

        #stats['inStepN'].append(inStepN)
        stats_list[countIMFs-1].inStepN = inStepN
        
        IMF[countIMFs-1, :] = h
        
        f = f - h
    
        #Find max-frequency contained in residual signal
        if ift: ttime.tic 
        N_pp, k_pp, maxmins_pp, diffMaxmins_pp = find_max_frequency(f,tol=tol, mode = opts.BCmode, method = opts.Maxmins_method)
        if ift: time_max_nu += ttime.get_toc
        
        #stats_list.append(stats)

    IMF = IMF[0:countIMFs, :]
    IMF = np.vstack([IMF, f[:]])
    stats_list = stats_list[:countIMFs]

    IMF = IMF*Norm1f # We scale back to the original values

    if ift: 
        ttime.total_elapsed(from_1st_start = True, hhmmss = True)
        print('imfs calculation took: %f (%.2f)' % (time_imfs,100* time_imfs / ttime._dttot)) 
        print('mask calculation took: %f (%.2f)' % (time_mask,100* time_mask / ttime._dttot)) 
        print('mask length calculation took: %f (%.2f)' % (time_max_nu,100* time_max_nu / ttime._dttot)) 

    return IMF, stats_list

