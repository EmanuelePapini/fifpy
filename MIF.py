
#from . import FIF_Python_v2_9 as FIFpy
from . import MIFpy

import sys

_path_=sys.modules[__name__].__file__[0:-6]
window_file = _path_+'prefixed_double_filter.mat'

#WRAPPER (FOR THE MOMENT)
def MIF_run(*args,**kwargs):
    
    return MIFpy.MIF_run(*args,**kwargs)


import numpy as np


class MIF():
    """
    python class of the Multidimensional Iterative Filtering (MIF) method  
    
    Calling sequence example

        #create the signal to be analyzed
        import numpy as np
        x = np.linspace(0,2*np.pi,100,endpoint=False)
        y = np.sin(2*x) + np.cos(10*x+2.3)
        
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

    Eventual custom settings (e.g. Xi, delta and so on) must be specified at the time of initialization
    (see __init__ below)

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
    """


    def __init__(self, delta=0.001, alpha=30, NumSteps=1, ExtPoints=3, NIMFs=200, \
                       MaxInner=200, Xi=1.6, MonotoneMaskLength=True, verbose = False):



        self.__version__=MIFpy.__version__
        self.options={'delta' : delta, 'alpha' : alpha, 'verbose' : verbose, \
                      'NumSteps' : NumSteps, 'ExtPoints' : ExtPoints, 'NIMFs' : NIMFs, \
                      'MaxInner' : MaxInner, 'MonotoneMaskLength' : MonotoneMaskLength,\
                      'Xi' : Xi}

        self.options = MIFpy.Settings(**self.options)

        self.MIFpy = MIFpy
   
        self.ancillary = {}


    def run(self, in_f, M=np.array([])):

        self.IMF, self.stats_list = MIFpy.MIF_run(in_f, M = M, options = self.options,window_file=window_file)




