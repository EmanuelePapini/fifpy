"""
 Iterative Filtering python package

 Dependencies : numpy, scipy

 Authors: 
    Python version: Emanuele Papini - INAF (emanuele.papini@inaf.it) 
    Original matlab version: Antonio Cicone - university of L'Aquila (antonio.cicone@univaq.it)

"""
from .IF_aux import *
__version__='8.4'



################################################################################
###################### Iterative Filtering main functions ######################
################################################################################

def Settings(**kwargs):
    """
    Sets the default options to be passed to IF
    options are set in the form of a dictionary.
    options input as key options in kwargs are added to the default dictionary
    or modify default options.
    WARNING: NO CHECK IS DONE ON THE VALIDITY OF THE INPUT
    
    parameters (optional)
    ---------------------
    stdout = False  
        if True, print output to stdout
    verbose = False 
        toggle verbosity level.
    timeit = False  
        Toggle timing of calculations
        
    delta = 0.001 
        StoppingCriterion1: threshold in the difference of 2Norm 
    ExtPoints=3 
        StoppingCriterion2: minimum number of extrema
    NIMFs=200 
        StoppingCriterion3: maximum number of IMF to be extracted
    MaxInner=200 
        StoppingCriterion4: maximum number of iterations
    
    Xi=1.6 
        stretching factor of the mask
    alpha='ave' 
        sets how to calculate the masklength from freq. distribution
    MonotoneMaskLength=True 
        sets if a monotone mask length is forced
    NumSteps=1 
        number of internal steps in IF loop between two FFTs
    BCmode = 'clip' 
        BCmode: boundary of the signal (if periodic is 'wrap')
    Maxmins_method = 'zerocrossing' 
        see Maxmins in IF_aux
    imf_method = 'fft' 
        select the numerical method for computation ('fft' or 'numba')
    MaxlogM = None 
        Maximum allowed mask length (If None then this value is 
        automatically set to the length of the timeseries.)

    """

    options = {}
    # General 
    options['stdout'] = False  #if True, print output to stdout
    options['verbose'] = False #toggle verbosity level.
    options['timeit'] = False  #Toggle timing of calculations
        
    # FIF   #SC == Stopping criterion (convergence)
    options['delta'] = 0.001 #SC1: threshold in the difference of 2Norm 
    options['ExtPoints']=3 #SC2: minimum number of extrema
    options['NIMFs']=200 #SC3: maximum number of IMF to be extracted
    options['MaxInner']=200 #SC4: maximum number of iterations
    
    options['Xi']=1.6 #stretching factor of the mask
    options['alpha']='ave' #sets how to calculate the masklength from freq. distribution
    options['MonotoneMaskLength']=True #sets if a monotone mask length is forced
    options['NumSteps']=1 #number of internal steps in IF loop between two FFTs
    options['BCmode'] = 'clip' #BCmode: boundary of the signal (if periodic is 'wrap')
    options['Maxmins_method'] = 'zerocrossing' #see Maxmins in IF_aux
    options['imf_method'] = 'fft' #numba #select the numerical method for computation
    options['MaxlogM'] = None # Maximum allowed mask length (If None then this value is
                              # automatically set to the length of the timeseries.
    for i in kwargs:
        if i in options.keys() : options[i] = kwargs[i] 
    return AttrDictSens(options)

#WRAPPER (version unaware. To be called by FIF_main.py) 
def FIF_run(x, options=None, M = np.array([]),**kwargs):
    
    if options is None:
        options = Settings()
    
    return IterativeFiltering(x,options,M=M,**kwargs)


def IterativeFiltering(f,options,M=np.array([]), window_mask=None, data_mask = None, nthreads = None):

    """
    Iterative Filtering python implementation (version 8) Parallel version
    adapted from MvFIF_v8.m
   
    Parameters
    ----------

    f : 1D array-like (size N)
        input timeseries
    options : dict
        dictionary containing the options of the IF decomposition (see Settings)
    
    Parameters (optional)
    ---------------------

    M : np.array 
        if not empty, the length of the mask for each imf is imposed by M
    
    window_mask : None or 1D-array
        window mask to be used to calculate the window function with variable length set
        by logM in the code (via interpolation)

    data_mask : None or boolean array of size x
        used to mask data that wont be used to determine the size of the window mask (LogM).
        if not None, then the indices where data_mask is True are not used to calculate the
        maximum frequency contained in the signal
    nthreads : int or None
        number of threads to be used in the numba implementation
    """
    ssend = '\r'
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
        #compute_imf = compute_imf_fft_adv
    elif opts.imf_method == 'numba': 
        compute_imf = compute_imf_numba

    MaxlogM = np.size(f) if opts.MaxlogM is None else opts.MaxlogM
    #loading master filter
    ift = opts.timeit
    if ift: 
        from . import time_1
        ttime = time_1.timeit()
        time_imfs = 0.
        time_max_nu = 0.
        time_mask = 0.
        ttime.tic

    if window_mask is None:
        MM = FKmask #get_window_file_path()
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
    f_pp = np.delete(f,data_mask) if data_mask is not None else f
    N_pp, k_pp, maxmins_pp, diffMaxmins_pp = find_max_frequency(f_pp,tol=tol, \
        mode = opts.BCmode, method = opts.Maxmins_method)
    if ift: time_max_nu += ttime.get_toc

    countIMFs = 0
    stats_list = np.recarray(opts.NIMFs,dtype=[('logM',int),('inStepN',int)])
    stats_list.logM = 0
    stats_list.inStepN = 0
    logM = 1 
    ### Begin Iterating ###
    while countIMFs < opts.NIMFs and k_pp >= opts.ExtPoints:
        countIMFs += 1
        
        h = f
        if 'M' not in locals() or np.size(M)<countIMFs:
            m = get_mask_length(opts,N_pp,k_pp,diffMaxmins_pp,logM,countIMFs)
        else:
            m = M[countIMFs-1]
        if N < 2*m+1: 
            if opts.verbose: 
                print('Mask length %d exceeds signal length %d. Finishing...'%(2*m+1,N))
            countIMFs -= 1
            break
        if logM > MaxlogM:
            if opts.verbose: print('Mask length exceeds Maximum allowed length, Finishing...')
            countIMFs -= 1
            break

        if countIMFs ==opts.NIMFs: ssend = '\n'
        if opts.stdout : print('IMF', countIMFs,' (%d/%d)'%(m,N), end=ssend)
        
        if opts.verbose:
            print('\n IMF # %1.0d   -   # Extreme points %5.0d\n' %(countIMFs,k_pp))
            print('\n  step #            SD             Mask length \n\n')

        stats_list[countIMFs-1].logM = int(m)
        logM = int(m)
        
        if ift: ttime.tic 
        a = get_mask_v1_1(MM, m,opts.verbose,tol)
        if ift: time_mask += ttime.get_toc
        #if the mask is bigger than the signal length, the decomposition ends.

        if ift: ttime.tic 
        h, inStepN, SD = compute_imf(h,a,opts)
        if ift: time_imfs += ttime.get_toc
        
        if inStepN >= opts.MaxInner:
            if opts.stdout : print('Max # of inner steps reached')

        stats_list[countIMFs-1].inStepN = inStepN
        
        IMF[countIMFs-1, :] = h
        
        f = f - h
    
        #Find max-frequency contained in residual signal
        if ift: ttime.tic 
        f_pp = np.delete(f,data_mask) if data_mask is not None else f
        N_pp, k_pp, maxmins_pp, diffMaxmins_pp = find_max_frequency(f_pp,tol=tol, \
            mode = opts.BCmode, method = opts.Maxmins_method)
        if ift: time_max_nu += ttime.get_toc
        

    IMF = IMF[0:countIMFs, :]
    IMF = np.vstack([IMF, f[:]])
    stats_list = stats_list[:countIMFs]

    IMF = IMF*Norm1f # We scale back to the original values
    if opts.stdout : print('',end='\n')
    if ift: 
        ttime.total_elapsed(from_1st_start = True, hhmmss = True)
        print('imfs calculation took: %f (%.2f)' % (time_imfs,100* time_imfs / ttime._dttot)) 
        print('mask calculation took: %f (%.2f)' % (time_mask,100* time_mask / ttime._dttot)) 
        print('mask length calculation took: %f (%.2f)' % (time_max_nu,100* time_max_nu / ttime._dttot)) 

    return IMF, stats_list

