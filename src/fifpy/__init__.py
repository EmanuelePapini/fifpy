"""
 Fast Iterative Filtering python package

 Dependencies : numpy, scipy, numba, pylab, attrdict

 Authors: 
    Python version: Emanuele Papini - INAF (emanuele.papini@inaf.it) 
    Original matlab version: Antonio Cicone - university of L'Aquila (antonio.cicone@univaq.it)
"""
# Monkey patch collections to fix incompatibility with attrdict
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))


from . import fif_tools as ftools

#import sys
import numpy as np
from copy import copy

from . import FIFpy
from . import MvFIFpy
from . import IFpy
from . import MvIFpy
from . import MIFpy

__version__ = ('FIF:'+FIFpy.__version__,'MvFIF:'+MvFIFpy.__version__,\
               'IF:'+IFpy.__version__,'MvIF:'+MvIFpy.__version__,\
               'MIF:'+MIFpy.__version__)

#_path_=sys.modules[__name__].__file__[0:-11]
#window_file = _path_+'prefixed_double_filter.mat'





class FIF():
    """

    Python class of the Fast Iterative Filtering (FIF) method  
    
    Calling sequence example

        #create the signal to be analyzed
        import numpy as np
        x = np.linspace(0,2*np.pi,100,endpoint=False)
        y = np.sin(2*x) + np.cos(10*x+2.3)
        
        #do the fifpy analysis
        import fifpy 

        fif=fifpy.FIF()

        fif.run(y)

        #plot the results
        import pylab as plt
        plt.ion()
        plt.figure()
        plt.plot(x,y,label='signal')
        [plt.plot(x,fif.data['IMC'][i,:],label = 'IMC#'+str(i)) for i in range(fif.data['IMC'].shape[0])]
        plt.legend(loc='best')
    
    WARNING: This class is not fully documented nor error proof.
    Should you need help, please contact Emanuele Papini (emanuele.papini@inaf.it) 

    Optional settings (e.g. Xi, delta and so on) must be specified at the time of initialization
    
    Parameters
    ---------- 
    delta : float (default 0.001)
        Stopping criterion based on the 2norm difference threshold between the nth 
        and the nth-1 high-pass-filtered signal in the iteration.
        Iteration stops if 2norm(nth) -2norm(nth-1) <=delta
    ExtPoints : int (default 3)
        Stopping criterion.
        Stops the IMCs extraction/calculation if the extrema (maxima+minima) in the 
        residual signal to be analyzed are <= ExtPoints
    NIMFs : int (default 200)
        Stopping criterion
        Maximum number of IMCs to be extracted.
    Xi : float (default 1.6)
        Stretching factor of the mask/window-function length
    alpha : str or int (default 'ave')
        sets how to calculate the (half) mask length of the IMCs
        int: 0 to 100: take as half mask length the "alpha" percentile of the distribution of
             distances between extrema found in the signal
        'ave': take as half mask length the number of points in the signal (N) divided by
            the number of extrema.
        'Almost_min': take the minimum between the 305h percentile of the distribution and 'ave'
    MaxInner : int (default 200)
        Stopping criterion. The maximum number of iterations allowed in the extraction of one IMC
    MonotoneMaskLength : bool (default True)
        If True, the mask length of the next IMC is forced to be 1.1*mask length of the current one if
        the mask length calculation of the next IMC returns a smaller length.
    NumSteps : int (default 1)
        number of steps to do in the iteration between the check of the 2norm difference (see delta)
    verbose : bool (default False)
        toggle verbosity

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
        """
        Run the decomposition
        
        Parameters
        ----------
        in_f : 1d array of float
            input signal to be decomposed
        M : list of int (optional)
            list of mask lengths to be used (optional). 
            NOTE: this will force the decomposition to use predefined mask lengths.
        wshrink : int
            number of points to exclude from the output (used to exclude, e.g. periodic extensions
            of the original signal).

        **kwargs: dict
            optional kwargs to be passed to self.FIFpy.FIF_run

        """
        self.data = {}
        
        self.data['IMC'], self.data['stats_list'] = self.FIFpy.FIF_run(in_f, M = M,\
            options = self.options,**kwargs)

        self.ancillary['wshrink'] = wshrink
        
        self.wsh = wshrink

    @property
    def input_timeseries(self):
        """
        return the input timeseries (including extension)
        """
        return np.sum(self.data['IMC'],axis=0)
    
    @property
    def IMC(self):
        """
        Return the extracted IMCs (excluding the extension)
        """
        return self.data['IMC'][:,self.wsh:-self.wsh] if self.wsh >0 else self.data['IMC'] 

    def get_freq_amplitudes(self, as_output = False, use_instantaneous_freq = True,  **kwargs):
        """
        Calculates average (integrated) instant frequencies and amplitudes of all IMCs

        Parameters
        ----------
        as_output : bool
            if set to True, then it returns inst. frequency and amplitude.
            If set to false, save the result in self.data['freqs'] and self.data['amps'].
        use_instantaneous_freq : bool
            use the instantaneous freq. to compute the average freq of the IMC
        
        the available **kwargs are the following kwargs from fifpy.fif_tools.IMC_get_freq_amplitudes
            dt : float (default = 1.) 
                time resolution (inverse of the sampling frequency) 
            resort: bool
                if true, imfs are resorted according to decreasing frequency
        """
        wsh = self.ancillary['wshrink']

        self.data['freqs'], self.data['amps'] = ftools.IMC_get_freq_amp(self.data['IMC'], \
            use_instantaneous_freq = use_instantaneous_freq, wshrink = wsh,  **kwargs)

        self.ancillary['get_freq_amplitudes'] = kwargs
        self.ancillary['get_freq_amplitudes']['use_instantaneous_freq'] = use_instantaneous_freq
        
        if as_output: return self.data['freqs'], self.data['amps']

    def orthogonalize(self,threshold = 0.6, only_nearest = True, **kwargs):
        """
        check orthogonality between IMCs and orthogonalize the set by aggregating non-orthogonal IMCs,
        i.e. IMCs for which

            <IMCs[i]*IMCs[j]> / sqrt(<IMCs[i]**2> * <IMCs[j]**2>) >= threshold

        where <...> is the inner product (i.e. the integral).

        Parameters
        ----------
        threshold : float 
            threshold parameter, must range from 0 to 1.
        only_nearest : bool
            if True, then only nearest IMF (if not orthogonal according to threshold) are aggregated.
        """
        if self.data['IMC'].shape[0] <3: #if shape==2 then only one imf was extracted
            return
        IMCs = self.data['IMC']
        imfs = ftools.orthogonalize(IMCs,threshold, only_nearest, **kwargs)
        self.ancillary['orthogonalized'] = True
        self.data['IMC'] = imfs


    def copy(self):
        return copy(self)



class MvFIF(FIF):
    """
    Python class for performing the Multivariate Fast Iterative Filtering decomposition. 
    
    (see Cicone and Pellegrino, IEEE Transactions on Signal Processing, vol. 70, pp. 1521-1531)

    WARNING: This class is not fully documented nor error proof.
    Should you need help, please contact Emanuele Papini (emanuele.papini@inaf.it) 
    
    Parameters
    ----------
    delta : float (default 0.001)
        Stopping criterion based on the 2norm difference threshold between the nth 
        and the nth-1 high-pass-filtered signal in the iteration.
        Iteration stops if 2norm(nth) -2norm(nth-1) <=delta
    ExtPoints : int (default 3)
        Stopping criterion.
        Stops the IMCs extraction/calculation if the extrema (maxima+minima) in the 
        residual signal to be analyzed are <= ExtPoints
    NIMFs : int (default 200)
        Stopping criterion
        Maximum number of IMCs to be extracted.
    Xi : float (default 1.6)
        Stretching factor of the mask/window-function length
    alpha : str or int (default 'ave')
        sets how to calculate the (half) mask length of the IMCs
        int: 0 to 100: take as half mask length the "alpha" percentile of the distribution of
             distances between extrema found in the signal
        'ave': take as half mask length the number of points in the signal (N) divided by
            the number of extrema.
        'Almost_min': take the minimum between the 305h percentile of the distribution and 'ave'
    MaxInner : int (default 200)
        Stopping criterion. The maximum number of iterations allowed in the extraction of one IMC
    MonotoneMaskLength : bool (default True)
        If True, the mask length of the next IMC is forced to be 1.1*mask length of the current one if
        the mask length calculation of the next IMC returns a smaller length.
    NumSteps : int (default 1)
        number of steps to do in the iteration between the check of the 2norm difference (see delta)
    verbose : bool (default False)
        toggle verbosity
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
    
    @property
    def IMC(self):
        """
        Return the extracted IMCs (excluding the extension)
        """
        return self.data['IMC'][:,:,self.wsh:-self.wsh] if self.wsh >0 else self.data['IMC'] 

    def get_freq_amplitudes(self, as_output = False, use_instantaneous_freq = True,  **kwargs):
        """
        
        Calculates the instantaneous frequencies and amplitudes of the IMCs.
        
        Parameters
        ----------
        as_output : bool
            if set to True, then it returns inst. frequency and amplitude.
            If set to false, save the result in self.data['freqs'] and self.data['amps'].
        use_instantaneous_freq : bool
            use the instantaneous freq. to compute the average freq of the IMC
        
        the available **kwargs are the following kwargs from self.fif_tools.IMC_get_freq_amplitudes
            dt : float (default = 1.) 
                time resolution (inverse of the sampling frequency) 
            resort : Bool (default = False)
                if true, frequencies and amplitudes are sorted frequency-wise
                
        """
        wsh = self.ancillary['wshrink']
        
        nimf,ndim,nt = self.data['IMC'].shape
        freqs = np.zeros((ndim,nimf))
        amps = np.zeros((ndim,nimf))
        for i in range(ndim):
            freqt,ampt = ftools.IMC_get_freq_amp(self.data['IMC'][:,i,:], \
                     use_instantaneous_freq = use_instantaneous_freq, wshrink = wsh, \
                     **kwargs)
            freqs[i] = freqt
            amps[i] = ampt
        self.data['freqs'] = freqs 
        self.data['amps'] = amps
        self.ancillary['get_freq_amplitudes'] = kwargs
        self.ancillary['get_freq_amplitudes']['use_instantaneous_freq'] = use_instantaneous_freq
        
        if as_output: return self.data['freqs'], self.data['amps']

    def orthogonalize(self,threshold = 0.6, only_nearest = True, **kwargs):
        """
        check orthogonality between IMCs and orthogonalize the set by aggregating non-orthogonal IMCs,
        i.e. IMCs for which

            <IMCs[i]*IMCs[j]> / sqrt(<IMCs[i]**2> * <IMCs[j]**2>) >= threshold

        where <...> is the inner product (i.e. the integral).

        The procedure is performed on the single channels of the Mv signal, if one of them is found no to 
        be orthogonal, e,g, IMCs[i,k] and IMCs[j,k], then all ith and jth IMCs (all k ) are aggregated.
        
        Parameters
        ----------
        threshold : float 
            threshold parameter, must range from 0 to 1.
        only_nearest : bool
            if True, then only nearest IMF (if not orthogonal according to threshold) are aggregated.
        """
        
        if self.data['IMC'].shape[0] <3: #if shape==2 then only one imf was extracted
            return
        IMCs = self.data['IMC']
        imfs = ftools.orthogonalize_MvFIF(IMCs,threshold, only_nearest, **kwargs)
        self.ancillary['orthogonalized'] = True
        self.data['IMC'] = imfs

class IF(FIF):
    """
    Advanced class for Iterative Filtering decompostion. (Fast version using FFT to make convolutions),
    
    It contains all the core features of the IF class plus some methods
    to perform statistics over the computed IMCs.
    
    N.B. This is not FIF, but it is as fast as FIF.
    """

    def __init__(self, **kwargs):
        """
        initialize Iterative Filtering Class.
        For kwargs options please look at fifpy.IFpy.Settings()
        """


        self.__version__=IFpy.__version__
        self.options = IFpy.Settings(**kwargs)


        self.FIFpy = IFpy
   
        self.ancillary = {}


    
    def run(self, in_f, M=np.array([]), wshrink = 0, preprocess = None, get_output = False,\
            data_mask = None,npad_raisedcos = None):
        """
        Run the decomposition
        
        Parameters
        ----------
        
        in_f : 1d array of float (size(N))
            input signal to be decomposed
        
        preprocess : str
            allowed values : 'make-periodic', 'extend-periodic', None
            'make-periodic': make in_f periodic prior running IF decomposition by means of a raised 
                             cosine windowing of length wshrink. The non-windowed part of in_f is
                             in_f[wshrink:-wshrink]
            'extend-periodic': extend in_f to make it periodic prior running IF decomposition by means of a raised 
                             cosine windowing of length wshrink, applied on an extend version of in_f obtained 
                             using the fifpy.arrays.extend_periodic method.
            None: self explaining
        M : list of int (optional)
            list of mask lengths to be used (optional). 
            NOTE: this will force the decomposition to use predefined mask lengths.
        
        wshrink : int
            number of points to exclude from the output (used to exclude, e.g. periodic extensions
            of the original signal).
        get_output : bool
            if True, it returns the output of the decomposition instead of saving it in self.data
        data_mask :1D array of bool (size N)
            if input, then the points where data_mask == True are excluded from the calculation
            of the mask length. This is used, e.g. to deal with gaps that may be present in the data
            which must be filled before starting the decomposition.
        npad_raisedcos : int or sequence of ints or None
            number of points (from left and right boundary) where to apply the raised cosine.
            If None, then npad_raisedcos = wshrink if 'make-periodic' or 'extend-periodic' are selected.

        """
        if preprocess == 'make-periodic':
            if self.options.verbose: print('\nmaking input signal periodic...')
            
            from .arrays import make_periodic
            
            if wshrink == 0 : wshrink = in_f.size//4 
            
            in_f = make_periodic(in_f,wshrink)
        
        elif preprocess == 'extend-periodic':
            if self.options.verbose: print('\nextending input signal (asymmetric-periodic)...')
            
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
    Advanced class for Multivariate IF decompostion.
    It contains all the core features of the MvFIF class plus some methods
    to perform statistics over the computed IMCs.
    """

    def __init__(self, **kwargs):
        """
        initialize MultiVariate Iterative Filtering Class.
        For kwargs options please look at the Settings method in fifpy.MvIFpy
        """


        self.__version__=MvIFpy.__version__
        self.options = MvIFpy.Settings(**kwargs)


        self.FIFpy = MvIFpy
   
        self.ancillary = {}


    def run(self, in_f, M=np.array([]), wshrink = 0, preprocess = None, get_output = False,\
            data_mask = None, npad_raisedcos = None):
        """
        Parameters
        ----------
        
        in_f : 1d array of float (size(N))
            input signal to be decomposed
        
        preprocess : str
            allowed values : 'make-periodic', 'extend-periodic', None
        
        M : list of int (optional)
            list of mask lengths to be used (optional). 
            NOTE: this will force the decomposition to use predefined mask lengths.
        wshrink : int
            number of points to exclude from the output (used to exclude, e.g. periodic extensions
            of the original signal).
        get_output : bool
            if True, it returns the output of the decomposition instead of saving it in self.data
        data_mask :1D array of bool (size N)
            if input, then the points where data_mask == True are excluded from the calculation
            of the mask length. This is used, e.g. to deal with gaps that may be present in the data
            which should be filled before starting the decomposition.
        
        npad_raisedcos : int or sequence of ints or None
            number of points (from left and right boundary) where to apply the raised cosine.
            If None, then npad_raisedcos = wshrink if 'make-periodic' or 'extend-periodic' are selected.

        """
        silent = self.options['silent']
        D,N = np.shape(in_f)
        if preprocess == 'make-periodic':
            if not silent: print('\nmaking input signal periodic...')
            from .arrays import make_periodic
            
            if wshrink == 0 : wshrink = in_f.size//4 
            
            out_f = np.zeros((D,N)) 
            for iD in range(D):    
                out_f[iD] = make_periodic(in_f[iD],wshrink)
        elif preprocess == 'extend-periodic':
            if not silent: print('\nextending input signal (asymmetric-periodic)...')

            from .arrays import extend_signal
            
            if wshrink == 0 : wshrink = in_f.shape[-1]//2 
            
            ff = extend_signal(in_f[0],wshrink,npad_raisedcos = npad_raisedcos) 
            out_f = np.zeros((D,ff.size))
            out_f[0] = ff
            for iD in range(1,D):
                out_f[iD] = extend_signal(in_f[iD],wshrink,npad_raisedcos = npad_raisedcos) 

            if data_mask is not None:
                data_mask = extend_signal(data_mask,wshrink,mode='reflect')
        
        else:
            if preprocess is not None: Warning('wrong input in keyword argument preprocess. Falling back to None')
            out_f = in_f

        MvFIF.run(self,out_f, M=M, wshrink= wshrink, data_mask = data_mask)

        if get_output == True:
            return self.data['IMC'][:,wshrink:-wshrink]

class MIF():
    """
    python class of the Multidimensional Iterative Filtering (MIF) method  
    
    Calling sequence example

        #create the signal to be analyzed
        import numpy as np
        n=512

        x = np.linspace(0,6*np.pi,n,endpoint=False)
        y2 = np.sin(2*x[:,None])*np.cos(2*x[None,:]+0.2) 
        y1 = np.cos(10*x[:,None]+2.3)*np.sin(11*x[None,:])
        y = y1+y2
        
        #do the MIF analysis
        import MIF
    
        mif_object=MIF.MIF()

        mif_object.run(y)

        #plot the results
        import pylab as plt
        plt.ion()
        plt.figure()
        plt.plot(x,y,label='signal')
        [plt.plot(x,mif_object.IMF[i,:],label = 'IMF#'+i.str()) for i in range(a.IMF.shape[0])]
        plt.legend(loc='best')

    Custom settings (e.g. Xi, delta and so on) must be specified at the time of initialization
    (see fifpy.MIFpy.Settings )

    """

    def __init__(self, **kwargs):
        """
        initialize Multidimensional Iterative Filtering Class.
        For kwargs options please look at the Settings method in fifpy.MIFpy
        """


        self.options = MIFpy.Settings(**kwargs)

        self.FIFpy = MIFpy
   
        self.ancillary = {}


    def run(self, in_f, M=np.array([]),**kwargs):
        """
        Parameters
        ----------
        
        in_f : 1d array of float (size(N))
            input signal to be decomposed
        
        M : list of int (optional)
            list of mask lengths to be used (optional). 
            NOTE: this will force the decomposition to use predefined mask lengths.

        """

        self.data = {}
        
        self.data['IMC'], self.data['stats_list'] = self.FIFpy.FIF_run(in_f, M = M,\
            options = self.options,**kwargs)

    @property
    def input_field(self):
        """
        return the input field 
        """
        return np.sum(self.data['IMC'],axis=0)
    @property
    def IMC(self):
        """
        return the result of the decomposition
        """
        return self.data['IMC']#[:,self.wsh:-self.wsh] if self.wsh >0 else self.data['IMC'] 
    

    def get_freq_amplitudes(self, as_output = False, use_instantaneous_freq = True,nsamples = None,  **kwargs):
        """
        Calculates the instantaneous frequencies and amplitudes of the IMCs.
        
        see fif_tools.IMC_get_freq_amplitudes for a list of **kwargs.
        
        Parameters
        ----------
        as_output : bool
            self-explaining
        use_instantaneous_freq = True : bool
            use the instantaneous freq. to compute the average freq of the IMC
        
        the available **kwargs should be
            dt : float (default = 1.) 
                grid resolution (inverse of the sampling frequency)
                WARNING! It is assumed that 2D grids have same resolution in x and y
            resort : Bool (default = False)
                if true, frequencies and amplitudes are sorted frequency-wise
        see fif_tools.IMC_get_freq_amp_MIF for a list of **kwargs
                
        """
        
        if nsamples is None: nsamples = self.options.Maxmins_samples
        
        self.data['freqs'], self.data['amps'] = \
            ftools.IMC_get_freq_amp_MIF(self.data['IMC'], \
            use_instantaneous_freq = use_instantaneous_freq, nsamples = nsamples,  **kwargs)

        self.ancillary['get_freq_amplitudes'] = kwargs
        self.ancillary['get_freq_amplitudes']['use_instantaneous_freq'] = use_instantaneous_freq
        
        if as_output: return self.data['freqs'], self.data['amps']

    def orthogonalize(self,threshold = 0.6, only_nearest = True, **kwargs):
        """
        check orthogonality between IMCs and orthogonalize the set by aggregating non-orthogonal IMCs,
        i.e. IMCs for which

            <IMCs[i]*IMCs[j]> / sqrt(<IMCs[i]**2> * <IMCs[j]**2>) >= threshold

        where <...> is the inner product (i.e. the integral).

        The procedure is performed on the single channels of the Mv signal, if one of them is found no to 
        be orthogonal, e,g, IMCs[i,k] and IMCs[j,k], then all ith and jth IMCs (all k ) are aggregated.
        
        Parameters
        ----------
        threshold : float 
            threshold parameter, must range from 0 to 1.
        only_nearest : bool
            if True, then only nearest IMF are aggregated (if not orthogonal according to threshold).
        """
        
        if self.data['IMC'].shape[0] <3: #if shape==2 then only one imf was extracted
            return
        IMCs = self.data['IMC']
        imfs = ftools.orthogonalize(IMCs,threshold, only_nearest, **kwargs)
        self.ancillary['orthogonalized'] = True
        self.data['IMC'] = imfs
