"""
 Multivariate Iterative Filtering python package

 Dependencies : numpy, scipy 
 Authors: 
    Python version: Emanuele Papini - INAF (emanuele.papini@inaf.it) 
    Original matlab version: Antonio Cicone - university of L'Aquila (antonio.cicone@univaq.it)

"""




from .IF_aux import *
__version__='8.4'

################################################################################
###################### Mv Iterative Filtering main functions ###################
################################################################################

def Settings(**kwargs):
    """
    Sets the default options to be passed to xFIF
    WARNING: NO CHECK IS DONE ON THE VALIDITY OF THE INPUT
    
    WRAPPED FROM: Settings_v3.m
    
    Parameters:
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
        If True, the mask length of the next IMCs is forced to be 1.1*mask length of the current one if
        the mask length calculation return a smaller length.
    NumSteps : int (default 1)
        number of steps to do in the iteration between the check of the 2norm difference (see delta)
    BCmode : str (default 'clip')
        boundary condition of the signal. Allowed values are the same of the 'mode' keyword in 
        scipy.signal.argrelextrema.
        'clip' : for non-periodic boundaries
        'wrap' : for periodic boundaries
    Maxmins_method  : str (default 'zerocrossing')
        method of calculation of number of extrema.
        'argrelextrema': compute maxima and minima using argrelextrema.
        'zerocrossing' : compute maxima and minima using zero crossing of 1st derivative.
    imf_method : str (default 'fft')
        'fft' : compute the convolution using FFTs in Fourier space
        'numba': compute the convolution in normal space using numba to speed up the calculations
    MaskLengthType : str (default 'amp')
        set the type of mask length to be calculated
        'amp': the mask is calculated based on the distribution of extrema in each signal
        'angle' : the mask is calculated using the extrema of the angle between the signals (deprecated)
                  This option is deprecated and not implemented in this version.
                  It is kept here for legacy with MvFIFpy.py
        The option 'amp' will be always used regardless.

    """

    options = {}
    # General 
    options['silent'] = False    
    options['verbose'] = False    
    options['timeit'] = False    
        
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
    options['MaskLengthType'] = 'amp'
    for i in kwargs:
        if i in options.keys() : options[i] = kwargs[i] 
    options['MaskLengthType'] = 'amp'
    return AttrDictSens(options)

def FIF_run(x, options=None, M = np.array([]),**kwargs):
    
    if options is None:
        options = Settings()
    
    return MvIF(x,options,M=M,**kwargs)


def MvIF(in_f,options,M=np.array([]), window_mask=None, data_mask = None, nthreads = None):

    """
    MultiVariate Iterative Filtering python implementation (version 8) Parallel version
    
    INPUT: x: array-like, shape(D,N)
        
        THE REST OF THE CRAZY INPUT IS AS USUAL :P

        the IF decomposition is along N, and the analysis is performed on
        D channels/signals at once omogenizing the window size to be the same
        i.e., it is a concurrent decomposition

    window_mask : None or 1D-like array
        window mask to be used in the low-pass filter.
        If None then the default Fokker-planck filter is used.
        NOTE: this mask is interpolated to the desired mask-length at each iteration, so it is
        strongly recomended to provide a highly-resolved window-mask.
    data_mask : None or boolean array of size x
        used to mask data that wont be used to determine the size of the window mask (LogM).
        TO BE IMPLEMENTED
    nthreads : int
        number of threads to be used if numba option was selected in options['imf_method']

    """
    opts = AttrDictSens(options)
    silent = opts.silent
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

    if opts.imf_method == 'fft': 
        compute_imf = compute_imf_fft
        #compute_imf = compute_imf_fft_adv
    elif opts.imf_method == 'numba': 
        compute_imf = compute_imf_numba

    if opts.MaskLengthType == 'amp': 
        if not silent: print('using amplitude to calculate mask')
        tol = 1e-18
    #loading master filter
    ift = opts.timeit
    if ift: 
        from . import time_1
        ttime = time_1.timeit()
        time_imfs = 0.
        time_max_nu = 0.
        time_mask = 0.
        ttime.tic

    MM = FKmask if window_mask is None else window_mask
    f = np.copy(in_f)
    if len(f.shape) > 2: 
        raise Exception('Wrong dataset, the signal must be a 2D array!')
    
    #setting up machinery
    D,N = f.shape
    IMF = np.zeros([opts.NIMFs+1, D, N])
    #normalizing signal such as the maximum is +-1
    Norm1f = np.max(np.abs(f),axis=1)
    f /= Norm1f[:,None]

    #NOW starting the calculation of the IMFs

    #Find max-frequency contained in signal
    if ift: ttime.tic
    if opts.MaskLengthType == 'amp': 
        k_pp=N
        for ic in range(D):
            N_ppt, k_ppt, maxmins_ppt, diffMaxmins_ppt = \
                find_max_frequency(f[ic],tol=tol, mode = opts.BCmode, method = opts.Maxmins_method)
            if k_ppt<k_pp:
                N_pp, k_pp, maxmins_pp, diffMaxmins_pp = N_ppt, k_ppt, maxmins_ppt, diffMaxmins_ppt   
                del N_ppt, k_ppt, maxmins_ppt, diffMaxmins_ppt
    
    if ift: time_max_nu += ttime.get_toc

    countIMFs = 0
    stats_list = np.recarray(opts.NIMFs,dtype=[('logM',int),('inStepN',int)])
    stats_list.logM = 0
    stats_list.inStepN = 0
    logM = 1 
    ### Begin Iterating ###
    while countIMFs < opts.NIMFs and k_pp >= opts.ExtPoints:
        countIMFs += 1
        if not silent: print('IMF', countIMFs)
        h = np.copy(f)
        if 'M' not in locals() or np.size(M)<countIMFs:
            m = get_mask_length(opts,N_pp,k_pp,diffMaxmins_pp,logM,countIMFs)
        else:
            m = M[countIMFs-1]
        
        if opts.verbose:
            print('\n IMF # %1.0d   -   # Extreme points %5.0d\n' %(countIMFs,k_pp))
            print('\n  step #            SD             Mask length \n\n')

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
        inStepN = 0
        for ic in range(D):
            if not silent: print('extracting IMF %d, CHANNEL %d(%d), mask: %d\n'%(countIMFs,ic,D,m)) 
            hic, inStepNic, SDic = compute_imf(h[ic],a,opts)
            h[ic] = hic
            inStepN = np.max([inStepNic,inStepN])
        if ift: time_imfs += ttime.get_toc
        
        if inStepN >= opts.MaxInner:
            if not silent: print('Max # of inner steps reached')

        stats_list[countIMFs-1].inStepN = inStepN
        
        IMF[countIMFs-1, :,:] = h
        
        f = f - h
        #Find max-frequency contained in residual signal
        #The max frequency is the minimum among the max frequencies
        #found in the different channels
        if ift: ttime.tic 
        if opts.MaskLengthType == 'amp': 
            k_pp=N
            for ic in range(D):
                N_ppt, k_ppt, maxmins_ppt, diffMaxmins_ppt = \
                    find_max_frequency(f[ic],tol=tol, mode = opts.BCmode, method = opts.Maxmins_method)
                if k_ppt<k_pp:
                    N_pp, k_pp, maxmins_pp, diffMaxmins_pp = N_ppt, k_ppt, maxmins_ppt, diffMaxmins_ppt   
                del N_ppt, k_ppt, maxmins_ppt, diffMaxmins_ppt
        if ift: time_max_nu += ttime.get_toc
        

    IMF = IMF[0:countIMFs+1] # the last is empty and will be filled with the residual
    IMF[countIMFs] = f #last element filled with the residual
    stats_list = stats_list[:countIMFs]

    IMF = IMF*Norm1f[None,:,None] # We scale back to the original values

    if ift: 
        ttime.total_elapsed(from_1st_start = True, hhmmss = True)
        print('imfs calculation took: %f (%.2f)' % (time_imfs,100* time_imfs / ttime._dttot)) 
        print('mask calculation took: %f (%.2f)' % (time_mask,100* time_mask / ttime._dttot)) 
        print('mask length calculation took: %f (%.2f)' % (time_max_nu,100* time_max_nu / ttime._dttot)) 

    return IMF, stats_list

