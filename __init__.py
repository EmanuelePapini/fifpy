"""
 Fast Iterative Filtering python package

 Dependencies : numpy, scipy, numba, pylab, attrdict

 Authors: 
    Python version: Emanuele Papini - INAF (emanuele.papini@inaf.it) 
    Original matlab version: Antonio Cicone - university of L'Aquila (antonio.cicone@univaq.it)
"""



from . import fif_tools as ftools

import sys
import numpy as np
from copy import copy

from . import FIF_v2_13 as FIFpy
from . import MvFIFpy
from . import IFpy
from . import MvIFpy


__version__ = ('FIF:'+FIFpy.__version__,'MvFIF:'+MvFIFpy.__version__,'IF:'+IFpy.__version__,'MvIF:'+MvIFpy.__version__)

_path_=sys.modules[__name__].__file__[0:-11]
window_file = _path_+'prefixed_double_filter.mat'





class FIF():
    """
    WARNING: This is an experimental version with minimal explanation.
    Should you need help, please contact Emanuele Papini (emanuele.papini@inaf.it) 

    Python class of the (Fast) Iterative Filtering (FIF) method  
    
    Calling sequence example

        #create the signal to be analyzed
        import numpy as np
        x = np.linspace(0,2*np.pi,100,endpoint=False)
        y = np.sin(2*x) + np.cos(10*x+2.3)
        
        #do the FIF analysis
        import fifpy as FIF
    
        fif=FIF.FIF()

        fif.run(y)

        #plot the results
        import pylab as plt
        plt.ion()
        plt.figure()
        plt.plot(x,y,label='signal')
        [plt.plot(x,fif.data['IMC'][i,:],label = 'IMC#'+str(i)) for i in range(fif.data['IMC'].shape[0])]
        plt.legend(loc='best')

    Eventual custom settings (e.g. Xi, delta and so on) must be specified at the time of initialization
    (see __init__ below)
    Original matlab header
    % It generates the decomposition of the signal f :
    %
    %  f = IMF(1,:) + IMF(2,:) + ... + IMF(K, :)
    %
    % where the last row in the matrix IMF is the trend and the other rows
    % are actual IMFs
    %
    %                                Inputs
    %
    %   f         Signal to be decomposed
    %
    %
    %   M         Mask length values for each Inner Loop
    %
    %                               Output
    %
    %   IMF       Matrices containg in row i the i-th IMF. The last row
    %              contains the remainder-trend.
    %
    %   logM      Mask length values used for each IMF
    %
    %   See also SETTINGS_IF_V1, GETMASK_V1, MAXMINS_v3_4, PLOT_IMF_V8.
    %
    %  Ref: A. Cicone, J. Liu, H. Zhou. 'Adaptive Local Iterative Filtering for 
    %  Signal Decomposition and Instantaneous Frequency analysis'. Applied and 
    %  Computational Harmonic Analysis, Volume 41, Issue 2, September 2016, 
    %  Pages 384-411. doi:10.1016/j.acha.2016.03.001
    %  ArXiv http://arxiv.org/abs/1411.6051
    %
    %  A. Cicone. 'Nonstationary signal decomposition for dummies'. 
    %  Chapter in the book: Advances in Mathematical Methods and High 
    %  Performance Computing. Springer, 2019
    %  ArXiv https://arxiv.org/abs/1710.04844
    %
    %  A. Cicone, H. Zhou. 'Numerical Analysis for Iterative Filtering with 
    %  New Efficient Implementations Based on FFT'
    %  ArXiv http://arxiv.org/abs/1802.01359
    %
    
    Init Parameters 


    """


    def __init__(self, delta=0.001, alpha=30, NumSteps=1, ExtPoints=3, NIMFs=200, \
                       MaxInner=200, Xi=1.6, MonotoneMaskLength=True, verbose = False):



        self.__version__=FIFpy.__version__
        self.options={'delta' : delta, 'alpha' : alpha, 'verbose' : verbose, \
                      'NumSteps' : NumSteps, 'ExtPoints' : ExtPoints, 'NIMFs' : NIMFs, \
                      'MaxInner' : MaxInner, 'MonotoneMaskLength' : MonotoneMaskLength,\
                      'Xi' : Xi}

        if self.__version__ == '2.13':
            self.options = FIFpy.Settings(**self.options)

        self.FIFpy = FIFpy
   
        self.ancillary = {}
    
    def run(self, in_f, M=np.array([]), wshrink = 0,**kwargs):

        self.data = {}
        
        self.data['IMC'], self.data['stats_list'] = self.FIFpy.FIF_run(in_f, M = M,\
            options = self.options,window_file=window_file,**kwargs)

        self.ancillary['wshrink'] = wshrink
        
        self.wsh = wshrink

    @property
    def input_timeseries(self):
        return np.sum(self.data['IMC'],axis=0)
    @property
    def IMC(self):
        return self.data['IMC'][:,self.wsh:-self.wsh] if self.wsh >0 else self.data['IMC'] 

    def get_inst_freq_amp(self,dt, as_output = False ):
        """
        get instantaneous frequencies and amplitudes of the IMCs
        if as_output is true, it returns the result instead of adding it to data
        THIS METHOD IS DEPRECATED SINCE IT DOESN'T USE wshrink.
        PLEASE USE get_freq_amplitudes instead
        """
        
        if as_output:
            return ftools.IMC_get_inst_freq_amp(self.data['IMC'],dt)
        
        self.data['IMC_inst_freq'], self.data['IMC_inst_amp'] = ftools.IMC_get_inst_freq_amp(self.data['IMC'],dt)


    def get_freq_amplitudes(self, as_output = False, use_instantaneous_freq = True,  **kwargs):
        """
        see fif_tools.IMC_get_freq_amplitudes for a list of **kwargs

        the available **kwargs should be
            dt = 1. : float 
                grid resolution (inverse of the sampling frequency) 
            resort = False : Bool
                if true, frequencies and amplitudes are sorted frequency-wise
            wshrink = 0 : int 
                only IMC[:,wshrink:-wshrink+1] will be used to compute freqs and amps.
                To use if one needs to throw away the part of the IMC that goes, e.g.,
                into the periodicization
            use_instantaneous_freq = True : bool
                use the instantaneous freq. to compute the average freq of the IMC
                
        """
        wsh = self.ancillary['wshrink']

        self.data['freqs'], self.data['amps'] = ftools.IMC_get_freq_amp(self.data['IMC'], \
                                                    use_instantaneous_freq = use_instantaneous_freq, wshrink = wsh,  **kwargs)

        self.ancillary['get_freq_amplitudes'] = kwargs
        self.ancillary['get_freq_amplitudes']['use_instantaneous_freq'] = use_instantaneous_freq
        
        if as_output: return self.data['freqs'], self.data['amps']



    def copy(self):
        return copy(self)



class MvFIF(FIF):
    """
    Python class for performing the Multivariate Fast Iterative Filtering decomposition. 
    
    (see Cicone and Pellegrino, IEEE Transactions on Signal Processing, vol. 70, pp. 1521-1531)

    This is an experimental version with minimal explanation.
    Should you need help, please contact Emanuele Papini (emanuele.papini@inaf.it) or Antonio Cicone (antonio.cicone@univaq.it)
    """

    def __init__(self, delta=0.001, alpha=30, NumSteps=1, ExtPoints=3, NIMFs=200, \
                       MaxInner=200, Xi=1.6, MonotoneMaskLength=True, verbose = False):


        self.__version__=FIFpy.__version__

        self.options={'delta' : delta, 'alpha' : alpha, 'verbose' : verbose, \
                      'NumSteps' : NumSteps, 'ExtPoints' : ExtPoints, 'NIMFs' : NIMFs, \
                      'MaxInner' : MaxInner, 'MonotoneMaskLength' : MonotoneMaskLength,\
                      'Xi' : Xi}

        self.FIFpy = MvFIFpy
   
        #contains ancillary data which keep trace of the processing done on the data
        self.ancillary = {}


    def get_inst_freq_amp(self,dt, as_output = False ):
        """
        get instantaneous frequencies and amplitudes of the IMCs
        if as_output is true, it returns the result instead of adding it to data
        """
        
        if as_output:
            return ftools.IMC_get_inst_freq_amp(np.squeeze(self.data['IMC'][:,0,:]),dt)
        
        self.data['IMC_inst_freq'], self.data['IMC_inst_amp'] = \
            ftools.IMC_get_inst_freq_amp(np.squeeze(self.data['IMC'][:,0,:]),dt)


    def get_freq_amplitudes(self, as_output = False, use_instantaneous_freq = True,  **kwargs):
        """
        see fif_tools.IMC_get_freq_amplitudes for a list of **kwargs

        the available **kwargs should be
            dt = 1. : float 
                grid resolution (inverse of the sampling frequency) 
            resort = False : Bool
                if true, frequencies and amplitudes are sorted frequency-wise
            wshrink = 0 : int 
                only IMC[:,wshrink:-wshrink+1] will be used to compute freqs and amps.
                To use if one needs to throw away the part of the IMC that goes, e.g.,
                into the periodicization
            use_instantaneous_freq = True : bool
                use the instantaneous freq. to compute the average freq of the IMC
                
        """
        wsh = self.ancillary['wshrink']

        self.data['freqs'], self.data['amps'] = ftools.IMC_get_freq_amp(np.squeeze(self.data['IMC'][:,0,:]), \
                     use_instantaneous_freq = use_instantaneous_freq, wshrink = wsh, \
                     **kwargs)

        self.ancillary['get_freq_amplitudes'] = kwargs
        self.ancillary['get_freq_amplitudes']['use_instantaneous_freq'] = use_instantaneous_freq
        
        if as_output: return self.data['freqs'], self.data['amps']

class IF(FIF):
    """
    Advanced class for IF decompostion.
    It contains all the core features of the IF class plus some methods
    to perform statistics over the computed IMCs.
    """

    def __init__(self, **kwargs):
        """
        initialize Iterative Filtering Class.
        For kwargs options please look at the Settings method in IF_v8_3e.py
        """


        self.__version__=IFpy.__version__
        self.options = IFpy.Settings(**kwargs)


        self.FIFpy = IFpy
   
        self.ancillary = {}


    
    def run(self, in_f, M=np.array([]), wshrink = 0, preprocess = None,get_output = False,\
            data_mask = None,npad_raisedcos = None):
        """
        Parameters
        ----------
        preprocess : str
            allowed values : 'make-periodic', 'extend-periodic', None
        """
        if preprocess == 'make-periodic':
            print('\nmaking input signal periodic...')
            from .arrays import make_periodic
            
            if wshrink == 0 : wshrink = in_f.size//4 
            
            in_f = make_periodic(in_f,wshrink)
        
        elif preprocess == 'extend-periodic':
            print('\nextending input signal (asymmetric-periodic)...')

            from .arrays import extend_signal
            
            if wshrink == 0 : wshrink = in_f.shape[-1]//2 
            
            in_f = extend_signal(in_f,wshrink,npad_raisedcos = npad_raisedcos) 
            if data_mask is not None:
                data_mask = extend_signal(data_mask,wshrink,mode='reflect')
        else:
            if preprocess is not None: Warning('wrong input in keyword argument preprocess. Falling back to None')

        FIF.run(self,in_f, M=M, wshrink= wshrink, data_mask = data_mask)

        if get_output == True:
            return self.data['IMC'][:,wshrink:-wshrink]

        
class MvIF(MvFIF):
    """
    Advanced class for IF decompostion.
    It contains all the core features of the IF class plus some methods
    to perform statistics over the computed IMCs.
    """

    def __init__(self, **kwargs):
        """
        initialize Iterative Filtering Class.
        For kwargs options please look at the Settings method in IF_v8_3e.py
        """


        self.__version__=MvIFpy.__version__
        self.options = MvIFpy.Settings(**kwargs)


        self.FIFpy = MvIFpy
   
        self.ancillary = {}


    def run(self, in_f, M=np.array([]), wshrink = 0, preprocess = None,get_output = False,\
            data_mask = None,npad_raisedcos = None):
        """
        Parameters
        ----------
        preprocess : str
            allowed values : 'make-periodic', 'extend-periodic', None
        """
        D,N = np.shape(in_f)
        if preprocess == 'make-periodic':
            print('\nmaking input signal periodic...')
            from .arrays import make_periodic
            
            if wshrink == 0 : wshrink = in_f.size//4 
            
            out_f = np.zeros((D,N)) 
            for iD in range(D):    
                out_f[iD] = make_periodic(in_f[iD],wshrink)
        elif preprocess == 'extend-periodic':
            print('\nextending input signal (asymmetric-periodic)...')

            from .arrays import extend_signal
            
            if wshrink == 0 : wshrink = in_f.shape[-1]//2 
            
            ff = extend_signal(in_f[0],wshrink,npad_raisedcos = npad_raisedcos) 
            out_f = np.zeros((D,ff.size))
            out_f[0] = ff
            for iD in range(1,D):
                out_f[iD] = extend_signal(in_f[i],wshrink,npad_raisedcos = npad_raisedcos) 

            if data_mask is not None:
                data_mask = extend_signal(data_mask,wshrink,mode='reflect')
        
        else:
            if preprocess is not None: Warning('wrong input in keyword argument preprocess. Falling back to None')
            out_f = in_f

        MvFIF.run(self,out_f, M=M, wshrink= wshrink, data_mask = data_mask)

        if get_output == True:
            return self.data['IMC'][:,wshrink:-wshrink]

