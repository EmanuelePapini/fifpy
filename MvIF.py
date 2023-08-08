"""
 Multivariate Iterative Filtering python package

 Dependencies : numpy, scipy, numba, joblib  //TO CHECK

 Authors: 
    Python version: Emanuele Papini - INAF (emanuele.papini@inaf.it) 
    Original matlab version: Antonio Cicone - university of L'Aquila (antonio.cicone@univaq.it)

"""




import time
#import pandas as pd
import timeit
from .IF_aux import *
__version__='8.3e'


################################################################################
########################## AUXILIARY FUNCTIONS #################################
################################################################################

################################################################################
###################### Mv Iterative Filtering main functions ###################
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
    options['MaskLengthType'] = 'amp'
    for i in kwargs:
        if i in options.keys() : options[i] = kwargs[i] 
    return AttrDictSens(options)

def IF_run(x, options=None, M = np.array([]),**kwargs):
    
    if options is None:
        options = Settings()
    
    return IF_v8_3e(x,options,M=M,**kwargs)


def MvIF(f,options,M=np.array([]), window_file=None, data_mask = None, nthreads = None):

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
        TO BE IMPLEMENTED
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

    if opts.imf_method == 'fft': 
        #compute_imf = compute_imf_fft
        compute_imf = compute_imf_fft_adv
    elif opts.imf_method == 'numba': 
        compute_imf = compute_imf_numba

    if opts.MaskLengthType == 'amp': 
        print('using amplitude to calculate mask')
        tol = 1e-18
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
    D,N = f.shape
    IMF = np.zeros([NIMFs+1, D, N])
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
        #print('IMF', countIMFs)
        
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
        inStepN = 0
        for ic in range(D):
            hic, inStepNic, SDic = compute_imf(h[ic],a,opts)
            h[ic] = hic
            inStepN = np.max([inStepNic,inStepN])
        if ift: time_imfs += ttime.get_toc
        
        if inStepN >= opts.MaxInner:
            print('Max # of inner steps reached')

        #stats['inStepN'].append(inStepN)
        stats_list[countIMFs-1].inStepN = inStepN
        
        IMF[countIMFs-1, :,:] = h
        
        f = f - h
    
        #Find max-frequency contained in residual signal
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
        
        #stats_list.append(stats)

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

