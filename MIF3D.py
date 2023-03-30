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
__version__='3e'

#WRAPPER (version unaware. To be called by IF.py) 
# def IF_run(*args,**kwargs):
#     return IF_v8_3e(*args,**kwargs)

#WRAPPER (version unaware. To be called by FIF.py) 

################################################################################
########################## AUXILIARY FUNCTIONS #################################
################################################################################
#class AttrDictSens(dict):
#    '''
#    A case-sensitive dictionary with access via item, attribute, and call
#    notations:
#
#        >>> d = AttrDict()
#        >>> d['Variable'] = 123
#        >>> d['Variable']
#        123
#        >>> d.Variable
#        123
#        >>> d.variable
#        123
#        >>> d('VARIABLE')
#        123
#    '''
#
#    def __init__(self, init={}):
#        dict.__init__(self, init)
#
#    def __getitem__(self, name):
#        try :
#            return super(AttrDictSens, self).__getitem__(name)
#        except:
#            raise AttributeError
#
#    def __setitem__(self, key, value):
#        return super(AttrDictSens, self).__setitem__(key, value)
#
#    __getattr__ = __getitem__
#    __setattr__ = __setitem__
#    __call__ = __getitem__

def get_window_file_path():
    import sys
    _path_=sys.modules[__name__].__file__[0:-11]
    return _path_+'/prefixed_double_filter.mat'

def fftconvolveND(f,ker, mode = 'same', BCmode = 'wrap'):
    """
    h_ave = fftconvolve2D(h,kernel,mode='same',BC)
    SO FAR only mode = 'same' and BCmode = 'wrap' are implemented
    maybe use pylab_convolve2D for direct convolution
"""
    if any([i<j for i,j in zip(f.shape,ker.shape)]):
        print('error, kernel shape cannot be larger than 2D array shape')
        return None
    m = tuple([-i//2 for i in ker.shape])
    axxs = tuple([i for i in range(len(m))])
    pads = [(0,f.shape[i]-ker.shape[i]) for i in range(len(f.shape))]
    kpad = np.pad(ker,pads)
    kpad = np.roll(kpad,m,axxs)
    return np.fft.irfftn(np.fft.rfftn(f)*np.fft.rfftn(kpad),s=f.shape) 

################################################################################
###################### Iterative Filtering aux functions #######################
################################################################################

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
        if mode == 'clip':
            dx = np.diff(x) #naive derivative
        elif mode == 'wrap':
            dx = np.diff(np.concatenate([x,x[:2]])) #naive derivative
        sgns = np.diff(np.sign(dx)) #location of maxmins: max/min where sgns = -2/+2
        extrema = np.where(sgns != 0)[0]
        if len(extrema) < 1: return None
        dx = np.abs(dx[1:][extrema])
        extrema = extrema[dx>tol]
        if len(extrema) < 1: return None
        extrema +=1
    else: raise Exception('WRONG INPUT METHOD!')

    return extrema.squeeze()
   

def find_max_frequencyND(f,nsamples = 1, **kwargs):
    """
    find extrema contained in f and returns
    N_pp,k_pp,maxmins,diffMaxmins

    Input
    -----
    f: numpy ndarray 
        input multidimensional field (2D,3D,nD)
    
    nsamples: int
        number of samples per direction along which to compute the extrema.
        Increasing this number will improve the statisticts.
    output:
    N_pp : int, f.shape
    k_pp : tuple (len of f.shape), number of extrema found
    diff_xyz: list (len of f.shape) of arrays of integer (size k_pp -1): distance between neighbohr extrema
    
    """
    nxyz = f.shape
    ksamples = np.array(nxyz)//nsamples
    dims = tuple([i for i in range(len(f.shape))])
    k_pp = []
    diffMaxmins_xyz = [[] for i in range(len(dims))]
    maxmins_xyz = [[] for i in range(len(dims))]
    diff_xyz = [[] for i in range(len(dims))]
    for j in range(len(nxyz)):
        for i in range(nsamples):
            jdims = np.roll(dims,j)
            jnxyz = np.roll(nxyz,j)
            jkn = np.roll(ksamples,j)[:-1]
            k0 = np.sum([i*jkn[idim]*np.prod(jnxyz[idim+1:]) for idim in range(len(jdims)-1)])
            k1 = k0+nxyz[jdims[-1]]
            ftrans = f.transpose(jdims) 
            
            maxmins_xyz[j].append(Maxmins( ftrans.flat[k0:k1],**kwargs))
            if len(maxmins_xyz[j][-1]) > 1: diff_xyz[j].append(np.diff(maxmins_xyz[j][-1])) 
        
    k_pp = [[maxmins_xyz[j][i].shape[0] for j in range(len(nxyz))] for i in range(nsamples)]

    if np.sum(k_pp) == 0.:
        print('No extrema detected')
        return None,None,None,None
    
    #getting maximum number of extrema detected along x and y
    k_pp = np.flipud(np.array(k_pp).max(axis=0))

    diff_xyz = [np.concatenate([i.flatten() for i in j if i.size > 0]) for j in diff_xyz]
    diff_xyz.reverse()
    N_pp = f.shape

    return N_pp, k_pp, diff_xyz


def get_mask_lengthND(options,N_pp,k_pp,diff_xyz,logM,countIMFs):
    #m = get_mask_length2D(opts,N_pp,k_pp,diffMaxmins_pp,logM,countIMFs)
    
    if isinstance(options.alpha,str):
    
        if options.alpha == 'ave': 
            mm = [2*np.round(N_pp[i]/k_pp[i]*options.Xi) for i in range(len(N_pp))]
        elif options.alpha == 'Almost_min': 
            mm = [2*np.min( [options.Xi*np.percentile(diff_xyz[i],30), np.round(N_pp[i]/k_pp[i]*options.Xi)]) for i in range(len(N_pp))]
        elif options.alpha == 'Median':
            mm = [2*np.round(np.median(diff_xyz[i])*options.Xi) for i in range(len(N_pp))]
        else:    
            raise Exception('Value of alpha not recognized!\n')
    
    else:
        mm = [2*np.round(options.Xi*np.percentile(diffM_xyz[i],options.alpha)) for i in range(len(N_pp))]
    
    if countIMFs > 1:
        for i in range(len(mm)):
            if mm[i] <= logM[i]:
                if options.verbose:
                    print('Warning mask length (%d) is decreasing at step %1d. ' % (i,countIMFs))
                if options.MonotoneMaskLength:
                    mm[i] = np.ceil(logM[i] * 1.1)
                    if options.verbose:
                        print('The old mask length (x) is %1d whereas the new one is forced to be %1d.\n' % (
                        logM[i], mm[i]))
                else:
                    if options.verbose:
                        print('The old mask length (x) is %1d whereas the new one is %1d.\n' % (logM[i], mm[i]))

    return [int(i) for i in mm]



def get_mask_3D_v3(w,k):
    """
    
    function A=get_mask_2D_v3(w,k)
     
      get the mask with length 2*k+1 x 2*k+1
      k must be integer
      w is the area under the curve for each bar
      A  the mask with length 2*k+1 x 2*k+1
    wrapped from FIF2_v3.m
    """
    #check if k tuple contains integers
    if not all([type(i) is int for i in k]):
        print('input mask not integer, making it so')
        k=tuple([int(i) for i in k])

    L=np.size(w)
    m=(L-1)/2;  #2*m+1 =L punti nel filtro
    w = np.pad(w,(0,(L-1)//2))
    shapeout = tuple([2*i+1 for i in k])
    A=np.zeros(shapeout)
    if all([i<=m for i in k]): # The prefixed filter contains enough points
        #distance matrix
        xx = np.arange(-k[0],k[0]+1)/k[0]
        yy = np.arange(-k[1],k[1]+1)/k[1]
        zz = np.arange(-k[2],k[2]+1)/k[2]
        dm = np.sqrt(xx[:,None,None]**2 + yy[None,:,None]**2 + zz[None,None,:]**2) #normalized distance from ellipse border
        s = (m-1)+L/2*dm
        t = s+2
        s2 = np.ceil(s) - s
        t1 = t - np.floor(t)
        for ix in range(2*k[0]+1):
            for iy in range(2*k[1]+1):
                for iz in range(2*k[2]+1):
                    A[ix,iy,iz] = np.sum(w[int(np.ceil(s[ix,iy,iz]))-1:int(t[ix,iy,iz])]) +\
                         s2[ix,iy,iz] * w[int(np.ceil(s[ix,iy,iz]))-1] + t1[ix,iy,iz]* w[int(t[ix,iy,iz])-1]
        A/=np.sum(A)
    else : #We need a filter with more points than MM, we use interpolation
        print('Need to write the code!')
        A=[]

    return A


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

def compute_imf2d_fft(f,a,options):
        

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
        h_ave = fftconvolve2D(h,kernel,mode='same', BCmode = 'wrap')
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

def MIF3D_run(x, options=None, M = np.array([]),**kwargs):
    
    if options is None:
        options = Settings()
    
    return MIF_v3e(x,options,M=M,**kwargs)


def MIF3D_v3(f,options,M=np.array([]), window_file=None, data_mask = None, nthreads = None):

    """
    Multidimensional Iterative Filtering python implementation 
    
    
    INPUT: f: array-like, shape(M,N)
        
        THE REST OF THE CRAZY INPUT IS AS USUAL :P

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
    tol = 1e-12 

    if opts.imf_method == 'fft': 
        compute_imf3D = compute_imf2d_fft
        #compute_imf2D = _compute_imf2d_fft_adv
    elif opts.imf_method == 'numba': 
        compute_imf3D = compute_imf2d_numba

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
    if len(f.shape) != 2: 
        raise Exception('Wrong dataset, the signal must be a 2D array!')
    
    #setting up machinery
    nx,ny = f.shape
    IMF = np.zeros([opts.NIMFs, nx,ny])
    #normalizing signal such as the maximum is +-1
    Norm1f = np.max(np.abs(f))
    f = f/Norm1f

    #NOW starting the calculation of the IMFs

    #Find max-frequency contained in signal
    if ift: ttime.tic 
    N_pp, k_pp, diffMaxmins_x, diffMaxmins_y = \
        find_max_frequency2D(f,opts['Maxmins_samples'],tol=tol, mode = opts.BCmode, method = opts.Maxmins_method)
    if ift: time_max_nu += ttime.get_toc

    countIMFs = 0
    stats_list = np.recarray(opts.NIMFs,dtype=[('logMx',int),('logMy',int),('inStepN',int)])
    stats_list.logM = 0
    stats_list.inStepN = 0
    logM = (1,1) 
    ### Begin Iterating ###
    while countIMFs < opts.NIMFs and all([ii >= opts.ExtPoints for ii in k_pp]):
        countIMFs += 1
        print('IMF', countIMFs)
        
        h = f
        if 'M' not in locals() or np.size(M)<countIMFs:
            m = get_mask_length2D(opts,N_pp,k_pp,diffMaxmins_x,diffMaxmins_y,logM,countIMFs)
        else:
            m = M[countIMFs-1]
            if type(m) is int : m = (m)*2 

        if opts.verbose:
            print('\n IMF # %1.0d   -   # Extreme points (%s\n' %(countIMFs,k_pp))
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
        N_pp, k_pp, diffMaxmins_x, diffMaxmins_y = \
            find_max_frequency2D(f,opts['Maxmins_samples'],tol=tol, mode = opts.BCmode, method = opts.Maxmins_method)
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

