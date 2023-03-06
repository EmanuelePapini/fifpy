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
from numba import jit

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
    return _path_+'prefixed_double_filter.mat'


################################################################################
###################### Iterative Filtering aux functions #######################
################################################################################
def Maxmins(x,tol,mode='open'):
    """
    Wrapped from: Maxmins_v3_8 in FIF_v2_13.m
    """
    @jit(nopython=True) 
    def maxmins_wrap(x,df, N,Maxs,Mins):

        h = 1
        while h<N and np.abs(df[h]/x[h]) <= tol:
            h = h + 1
   
        if h==N:
            return None, None

        cmaxs = 0
        cmins = 0
        c = 0
        N_old = N
        
        df = np.zeros(N+h)
        df[0:N] = x
        df[N:N+h] = x[1:h+1]
        for i in range(N+h-1):
            df[i] = df[i+1] - df[i]
        
        f = np.zeros(N+h-1)
        f[0:N] = x
        f[N:] = f[1:h]
        
        N = N+h
        #beginfor
        for i in range(h-1,N-2):
            if abs(df[i]*df[i+1]/f[i]**2) <= tol :
                if df[i]/abs(f[i]) < -tol:
                    last_df = -1
                    posc = i
                elif df[i]/abs(f[i]) > tol:
                    last_df = +1
                    posc = i
                elif df[i] == 0:
                    last_df = 0
                    posc = i

                c = c+1

                if df[i+1]/abs(f[i]) < -tol:
                    if last_df == 1 or last_df == 0:
                        cmaxs = cmaxs +1
                        Maxs[cmaxs] = (posc + (c-1)//2 +1)%N_old
                    c = 0
                
                if df[i+1]/abs(f[i]) > tol:
                    if last_df == -1 or last_df == 0:
                        cmins = cmins +1
                        Mins[cmins] = (posc + (c-1)//2 +1)%N_old
                    c = 0

            if df[i]*df[i+1]/f[i]**2 < -tol:
                if df[i]/abs(f[i]) < -tol and df[i+1]/abs(f[i]) > tol:
                    cmins  =cmins+1
                    Mins[cmins] = (i+1)%N_old
                    if Mins[cmins]==0:
                        Mins[cmins]=1
                    last_df=-1

                elif df[i]/abs(f[i]) > tol and df[i+1]/abs(f[i])  < -tol:
                    cmaxs = cmaxs+1
                    Maxs[cmaxs] = (i+1)%N_old
                    if Maxs[cmaxs] == 0:
                        Maxs[cmaxs]=1
            
                    last_df =+1

        if c>0:
            if cmins>0 and Mins[cmins] == 0 : Mins[cmins] = N
            if cmaxs>0 and Maxs[cmaxs] == 0 : Maxs[cmaxs] = N

        return Maxs[0:cmaxs], Mins[0:cmins]

    N = np.size(x)

    Maxs = np.zeros(N)
    Mins = np.zeros(N)
    
    df = np.diff(x)

    if mode == 'wrap':
        Maxs, Mins = maxmins_wrap(x,df,N,Maxs,Mins)
        if Maxs is None or Mins is None:
            return None,None,None

        maxmins = np.sort(np.concatenate((Maxs,Mins) ))
        
        if any(Mins ==0): Mins[Mins == 0] = 1
        if any(Maxs ==0): Maxs[Maxs == 0] = 1
        if any(maxmins ==0): maxmins[maxmins == 0] = 1

    return maxmins,Maxs,Mins


################################################################################
###################### Iterative Filtering main functions ######################
################################################################################

def Settings(**kwargs):
    """
    Sets the default options to be passed to FIF
    WARNING: NO CHECK IS DONE ON THE VALIDITY OF THE INPUT
    
    WRAPPED FROM: Settings_v3.m
    """

    options = {}
    # General 
    #options['saveEnd'] = 0
    #options['saveInter'] = 0
    options['verbose'] = False    
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
    options['MaskLengthType']='angle'
    options['BCmode'] = 'clip' #wrap

    for i in kwargs:
        if i in options.keys() : options[i] = kwargs[i] 
    return AttrDictSens(options)

def IF_run(x, options=None, M = np.array([]),**kwargs):
    
    if options is None:
        options = Settings()
    
    return IF_v8_3e(x,options,M=M,**kwargs)


def IF_v8_3e(f,options,M=np.array([]), window_file=None, data_mask = None, nthreads = 1,verbose = True):

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
    if verbose:
        print('running IF decomposition...')
        #if verbose:
        print('****IF settings****')
        [print(i,options[i]) for i in options]
        print('data_mask   : ', data_mask is not None )
    
    tol = 1e-18 



    f = np.asarray(f)
    if len(f.shape) > 1: 
        raise Exception('Wrong dataset, the signal must be a 1D array!')
    
    #loading master filter
    if window_file is None:
        window_file = get_window_file_path()
    MM = loadmat(window_file)['MM'].flatten()
    
    #setting up machinery
    N = f.size
    IMF = np.zeros([options.NIMFs, N])
    #normalizing signal such as the maximum is +-1
    Norm1f = np.max(np.abs(f))
    f = f/Norm1f

    #NOW starting the calculation of the IMFs

    #Find max-frequency contained in signal


