import numpy as np

from scipy.integrate import trapezoid as integrate

def aggregate_IMCs_MIF(imfs,freqs,freq_ranges,return_mean_freq = False):
    """
    WARNING: this function is deprecated and it is kept for legacy purposes only. 
    Use fifpy.fif_tools.orthogonalize instead
    
    Aggregate 2D IMF obtained from the MIF decomposition according to frequency range 
    given in input. The frequency range defines the limits: i.e. if frequency range 
    contains two frequencies [f1,f2] then it returns 3 aggregate imf: one contaning 
    frequencies bewteen 0 and f1,
    the second between f1 and f2, and the last one containing all freqs.
    >f2.

    Parameters
    ----------
    imfs : 3D array
        array of shape (nimfs,nx,ny) of the imfs extracted from MIF.
    freqs : array (shape: (nimfs,))
        frequencies associated to the imfs.
    freq_ranges : 1D array like
        frequency ranges for the aggregation of imfs.
    return_mean_freq : bool
        If True, then it also return the frequencies given by the average 
        frequency of the range.
    
    output
    ------
    agimfs : 3D array of shape (len(freq_ranges)+1,nx,ny)
        output aggregated imfs
    mean_freqs: 1D array like of shape (len(freq_ranges)+1,)
        array of new average frequencies.
    """
    agimfs=np.zeros((len(freq_ranges)+1,imfs.shape[1],imfs.shape[2]))
    mean_freqs = np.zeros(agimfs.shape[0])
    f0 = 0.
    ii = 0
    for ifr in freq_ranges:
        mm = (freqs<=ifr) * (freqs>f0)
        f0=ifr     
        if np.sum(mm) > 0 : 
            agimfs[ii,:,:] = np.sum(imfs[mm,:,:],axis=0)
            mean_freqs[ii] = np.mean(freqs[mm])
        ii+=1
    
    mm = (freqs>f0)
    
    if np.sum(mm) > 0 : 
        agimfs[ii,:,:] = np.sum(imfs[mm,:,:],axis=0)
        mean_freqs[ii] = np.mean(freqs[mm])
    
    return (agimfs, mean_freqs) if return_mean_freq else agimfs


def check_orthogonality(imfs,periodic=True, plot=False):
    """
    calculate the orthogonality (cross-correlation) matrix M between the input imfs

        M_ij = Corr(imfs[i],imfs[j])
    
    where Corr is the Crosscorrelation operator.

    Parameters
    ----------
    imfs : 3D array
        array of shape (nimfs,nx,ny) of the imfs extracted from MIF.
    periodic : bool
        set to True if imfs are periodic 
    plot : bool
        plot the result
    WARNING: WORKS ONLY FOR 1D/2D IMCs. MUST BE IMPLEMENTED TO WORK 
             WITH NDimensional IMCs.

    """

    ndim = len(imfs.shape[1:])

    nimf=imfs.shape[0]

    orto = np.zeros([nimf,nimf])
    
    if periodic :
        imfst = np.zeros( (nimf,) + tuple(np.asarray(imfs.shape[1:])+1) )
        if ndim == 1:
            imfst[:,0:-1] =imfs[...]
            imfst[:,-1] = imfs[:,0]
        elif ndim == 2:
            #THIS IS THE PART THAT MUST BE GENERALIZED TO ND
            imfst[:,0:-1,0:-1] =imfs[:,:,:]
            imfst[:,-1,0:-1] = imfs[:,-1,:]
            imfst[:,0:-1,-1] = imfs[:,:,-1]
            imfst[:,-1,-1] = imfs[:,0,0]
    
    else : imfst = imfs

    if ndim == 1:
        amps = np.asarray([np.sqrt(integrate(imfst[i]**2)) for i in range(nimf)])

        for i in range(nimf):
            for j in range(i+1):
                orto[i,j] = integrate(imfst[i] * imfst[j])/amps[i]/amps[j]
    elif ndim == 2:
        amps = np.asarray([np.sqrt(integrate(integrate(imfst[i]**2))) for i in range(nimf)])

        for i in range(nimf):
            for j in range(i+1):
                orto[i,j] = integrate(integrate(imfst[i] * imfst[j]))/amps[i]/amps[j]


    if plot:
        import pylab as plt
        from matplotlib import cm
        plt.ion()
        plt.figure(figsize=(7,5.67))
        plt.imshow(orto,cmap=cm.Blues, extent=[0.5,nimf+.5,nimf+.5,0.5])
        for j in range(nimf):
            for i in range(j+1):
                plt.text(i+1,j+1,'{:.2f}'.format( np.abs(orto[j,i])),ha='center',va='center',fontsize=11,color='darkorange')
                plt.xlabel(r'$j$',fontsize=14)
                plt.ylabel(r'$i$',fontsize=14)
                plt.title(r'$\langle \mathrm{IMF}_i,\mathrm{IMF}_j\rangle $')
        plt.show()
    return orto

def orthogonalize(imfs,threshold = 0.6, **kwargs):
    """
    
    Sum imfs whose correlation is above the Threshold (see check_orthogonality).

    Parameters
    ----------
    imfs : 1D+ND array
        array of shape (nimfs,...) of the imfs extracted from MIF.
    
    threshold: float
        orthogonality thershold: imfs for which Corr(imfs[i],imfs[j]) > threshold
        are aggregated/summed.
    **kwargs:
        auxiliary input to be passed to check_orthogonality()
    WARNING: TESTED ONLY FOR 2D IMCS. IT WONT WORK WITH ND IMCS, SINCE 
             check_orthogonality HAS NOT BEEN IMPLEMENTED YET TO WORK WITH ND 
             IMCS.

    """

    orto =check_orthogonality(imfs,**kwargs)
    ilow = np.arange(imfs.shape[0]-1)+1
    lowdiag = orto[ilow,ilow-1]
    imfst = imfs
    while lowdiag.max() >= threshold:
        i = lowdiag.argmax()
        imt = imfst[i+1,...]
        imfst = np.concatenate((imfst[0:i+1,...],imfst[i+2:,...]),axis=0)
        imfst[i,...] += imt
        orto =check_orthogonality(imfst,**kwargs)
        ilow = np.arange(imfst.shape[0]-1)+1
        lowdiag = orto[ilow,ilow-1]
         
    return imfst

##### SPECIFIC MvFIF tools #####
def check_orthogonality_MvFIF(imfs,periodic=True, plot=False,only_nearest = False):
    """
    calculate the orthogonality (cross-correlation) matrix M between the input IMCs "imfs" 
    as given in output by MvIF / MvFIF decomposition.

        M_ij = Corr(imfs[i],imfs[j])
    
    where Corr is the Crosscorrelation operator.

    WORKS ONLY FOR 1D multichannel IMCs as given in output by MvFIF and MvIF

    Parameters
    ----------
    imfs : 3D array
        array of shape (nchannels,nimfs,nx) of the imfs extracted from Mv(F)IF.
    periodic : bool
        set to True if imfs are periodic 
    plot : bool
        plot the result
    only_nearest : bool
        if True only near imfs are checked for orthogonality.

    """

    ndim = len(imfs.shape[2:])

    nimf=imfs.shape[0]
    nchan = imfs.shape[1]
    
    if periodic :
        imfst = np.zeros( (nimf,nchan) + tuple(np.asarray(imfs.shape[2:])+1) )
        if ndim == 1:
            imfst[:,:,0:-1] =imfs[...]
            imfst[:,:,-1] = imfs[:,:,0]
        elif ndim == 2:
            #THIS IS THE PART THAT MUST BE GENERALIZED TO ND
            imfst[:,:,0:-1,0:-1] =imfs[:,:,:,:]
            imfst[:,:,-1,0:-1] = imfs[:,:,-1,:]
            imfst[:,:,0:-1,-1] = imfs[:,:,:,-1]
            imfst[:,:,-1,-1] = imfs[:,:,0,0]
    
    else : imfst = imfs

    if ndim == 1:
        amps = np.asarray([np.sqrt(integrate(imfst[i]**2)) for i in range(nimf)])

        if only_nearest:
            orto = np.zeros([nimf-1,nchan])
            for i in range(nimf-1):
                j = i+1
                ints = integrate(imfst[i] * imfst[j])
                orto[i] = ints/amps[i]/amps[j]
        else:
            orto = np.zeros([nimf,nimf,nchan])

            for i in range(nimf):
                for j in range(i+1):
                    ints = integrate(imfst[i] * imfst[j])
                    orto[i,j] = ints/amps[i]/amps[j]
        
        orto = orto.max(axis=-1)
    
    elif ndim == 2:
        raise Exception('NOT IMPLEMENTED FOR 2D SIGNALS')
        amps = np.asarray([np.sqrt(integrate(integrate(imfst[i]**2))) for i in range(nimf)])

        for i in range(nimf):
            for j in range(i+1):
                orto[i,j] = integrate(integrate(imfst[i] * imfst[j]))/amps[i]/amps[j]


    if plot:
        import pylab as plt
        from matplotlib import cm
        plt.ion()
        plt.figure(figsize=(7,5.67))
        plt.imshow(orto,cmap=cm.Blues, extent=[0.5,nimf+.5,nimf+.5,0.5])
        for j in range(nimf):
            for i in range(j+1):
                plt.text(i+1,j+1,'{:.2f}'.format( np.abs(orto[j,i])),ha='center',va='center',fontsize=11,color='darkorange')
                plt.xlabel(r'$j$',fontsize=14)
                plt.ylabel(r'$i$',fontsize=14)
                plt.title(r'$\langle \mathrm{IMF}_i,\mathrm{IMF}_j\rangle $')

    return orto

def orthogonalize_MvFIF(imfs,threshold = 0.6, only_nearest = True, **kwargs):
    """
    Sum imfs whose correlation is above the Threshold (see check_orthogonality_MvFIF).

    TESTED ONLY FOR 1D multichannel IMCS. 
    
    Parameters
    ----------
    imfs : 3D array of shape (nimfs,nchannels,nx)
        array of imcs as obtained from Mv(F)IF.
    threshold: float
        orthogonality thershold: imfs for which Corr(imfs[i],imfs[j]) > threshold
        are aggregated/summed.
    
    Output
    ------
    imfst: 3D array of aggregated imfs
    """

    orto =check_orthogonality_MvFIF(imfs,only_nearest = only_nearest, **kwargs)
    ilow = np.arange(imfs.shape[0]-1)+1
    lowdiag = orto if only_nearest else orto[ilow,ilow-1]
    imfst = imfs
    jjj=0
    while lowdiag.max() >= threshold:
        #print(jjj)
        #jjj+=1
        #print(imfst.shape)
        i = lowdiag.argmax()
        imt = imfst[i+1,...]
        imfst = np.concatenate((imfst[0:i+1,...],imfst[i+2:,...]),axis=0)
        imfst[i,...] += imt
        orto =check_orthogonality_MvFIF(imfst,only_nearest = only_nearest,**kwargs)
        ilow = np.arange(imfst.shape[0]-1)+1
        lowdiag = orto if only_nearest else orto[ilow,ilow-1]
         
    return imfst



def IMC_get_freq_amp(IMF, dt = 1, resort = False, wshrink = 0, use_instantaneous_freq = True):
    """
    Compute amplitude and average frequency of a set of IMCs.
    Parameters
    ----------
    IMF: 2D-array
        array containing IMCs as calculated from (F)IF.
    dt : float
        time/space resolution of the imfs
    resort: bool
        if true, imfs are resorted according to decreasing frequency
    use_instantaneous_freq : bool
        if True, then it uses the instantaneous frequency to calculate the average frequency of the IMC,
        if False, the average freq is found by counting maxima and minima in the IMC

    wshrink : int (positive)
        if >0, then only the central part of the IMC[:,wshrink:-wshrink] is used to calculate the amplitude
        This should be used when periodic extension is used or when windowing is used.

    """


    nimfs,nx = IMF.shape

    if use_instantaneous_freq:
        imf_if,imf_ia = IMC_get_inst_freq_amp(IMF,dt)

        freq = [np.sum(ifreq[0+wshrink:nx-wshrink]*iamp[0+wshrink:nx-wshrink])/np.sum(iamp[0+wshrink:nx-wshrink]) \
                for ifreq,iamp in zip(imf_if,imf_ia)]

    else:
        npeaks = np.array([np.size(Maxmins_v3_6(iimf)) for iimf in IMF])
        freq = npeaks/(2*nx*dt)
    
    amp0 = np.sqrt(integrate(IMF[:,0+wshrink:nx-wshrink]**2)*dt) 

    if resort:
        kf = np.argsort(freq)
        freq = freq[kf]
        amp0 = amp0[kf]

    return np.array(freq),np.array(amp0)

def IMC_get_inst_freq_amp(IMF,dt):
    """
    # Produces the istantaneous frequency and amplitude of a set of imfs.

    """
    fs = 1/dt #sampling frequency
    M0, N0 = np.shape(IMF)  #M0 = nimf, N0 = nt

    #arrays of inst. freqs and amplitudes
    IMF_iA = np.zeros((M0,N0))
    IMF_iF = np.zeros((M0,N0))
    
    min_iF = np.zeros(M0) 
    max_iF = np.zeros(M0) 
   
    zci = lambda v: np.find(np.diff(np.sign(v)))
    
    for i in range(M0):
        
        maxmins = Maxmins_v3_6(IMF[i,:])
        
        if np.size(maxmins) >= 2:
            temp_val = fs/(2*np.diff(maxmins))
            max_iF[i] = temp_val.max()
            min_iF[i] = temp_val.min()

            if maxmins[0] == 0 and maxmins[-1] == N0-1:
                IMF_iA[i] = np.interp(np.linspace(0,N0-1,N0), maxmins, abs(IMF[i, maxmins]))
                IMF_iF[i] = np.interp(np.linspace(0,N0-1,N0), maxmins, np.concatenate([temp_val, [temp_val[-1]]]))
            
            elif maxmins[0]!=0 and maxmins[-1]!=N0-1:

                dummy = np.concatenate([[0], maxmins, [N0-1]])
                IMF_iA[i] = np.interp(np.arange(float(N0)), dummy, abs(IMF[i, dummy]))
                IMF_iF[i] = np.interp(np.arange(float(N0)), dummy, np.concatenate([[temp_val[0]], temp_val, [temp_val[-1]]*2 ]))
            
            elif maxmins[0]!=0 and maxmins[-1]==N0-1:
                dummy = np.concatenate([[0], maxmins])
                IMF_iA[i] = np.interp(np.arange(float(N0)), dummy, abs(IMF[i, dummy]))
                IMF_iF[i] = np.interp(np.arange(float(N0)), dummy, np.concatenate([[temp_val[0]], temp_val, [temp_val[-1]]]))
            else:
                dummy = np.concatenate([maxmins,[N0-1]])
                IMF_iA[i] = np.interp(np.arange(float(N0)), dummy, abs(IMF[i, dummy]))
                IMF_iF[i] = np.interp(np.arange(float(N0)), dummy, np.concatenate([temp_val, [temp_val[-1]]*2]))


    return IMF_iF, IMF_iA




def Maxmins_v3_6(x, mode = 'wrap'):
    """
    find relative maxima and minima in a 1D signal x
    """
    from scipy.signal import argrelextrema
    maxima = argrelextrema(x, np.greater, mode = mode)
    minima = argrelextrema(x, np.less, mode = mode)

    extrema = np.sort(np.concatenate((maxima, minima), axis=1))

    return extrema.squeeze()



#MIF TOOLS
def IMC_get_freq_amp_MIF(IMF, dt = 1, resort = False, wshrink = 0, use_instantaneous_freq = True, nsamples = 4):
    """
    Compute amplitude and average frequency of a set of IMCs.
    Parameters
    ----------
    IMF: 3D-array
        array containing IMCs as calculated from MIF.
    dt : float
        time/space resolution of the imfs
    resort: bool
        if true, imfs are resorted according to decreasing frequency
    use_instantaneous_freq : bool
        if True, then it uses the instantaneous frequency to calculate the average frequency of the IMC,
        if False, the average freq is found by counting maxima and minima in the IMC

    wshrink : int (positive)
        if >0, then only the central part of the IMC[:,wshrink:-wshrink] is used to calculate the amplitude
        This should be used when periodic extension is used or when windowing is used.

    nsamples : int (positive)
        number of slices (along x and y) to use for the calculation of the frequency.
    """

    from random import randrange

    nimfs,nx,ny = IMF.shape

    if use_instantaneous_freq:
        freqx = []
        for ix in range(nsamples):
            imf_if,imf_ia = IMC_get_inst_freq_amp(IMF[:,randrange(nx),:].squeeze(),dt)

            freq = [np.sum(ifreq[0:nx]*iamp[0:nx])/np.sum(iamp[0:nx]) \
                    for ifreq,iamp in zip(imf_if,imf_ia)]
            freqx.append(freq)
        freqx=np.mean(np.asarray(freqx),axis=0)
        freqy = []
        for iy in range(nsamples):
            imf_if,imf_ia = IMC_get_inst_freq_amp(IMF[:,:,randrange(ny)].squeeze(),dt)

            freq = [np.sum(ifreq*iamp)/np.sum(iamp) \
                    for ifreq,iamp in zip(imf_if,imf_ia)]
            freqy.append(freq)
        freqy=np.mean(np.asarray(freqy),axis=0)
        freq = np.mean(np.asarray([freqx,freqy]),axis=0)
    else:
        npeaksx = np.array([np.mean([np.size(Maxmins_v3_6(iimf[randrange(nx),:].flatten())) for it in range(nsamples)]) for iimf in IMF])
        npeaksy = np.array([np.mean( [np.size(Maxmins_v3_6(iimf[:,randrange(ny)].flatten())) for it in range(nsamples)]) for iimf in IMF])
        freqx = npeaksx/(2*nx*dt)
        freqy = npeaksy/(2*ny*dt)
   
        freq = np.mean(np.asarray([freqx,freqy]),axis=0)
    
    amp0 = np.sqrt(integrate(integrate(IMF[:]**2)))*dt 
    if resort:
        kf = np.argsort(freq)
        freq = freq[kf]
        amp0 = amp0[kf]

    return np.array(freq),amp0

