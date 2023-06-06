# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:01:41 2016

@author: Emanuele Papini

This module contains all basic tools (functions /class) I developed
    
"""

import numpy as np



def make_periodic(sig, npoints, window_function = 'raised cosine'):
    """
    """
    
    nn = np.size(sig)
    
    if type(npoints) is int:
        
        if npoints > nn//2: 
            raise ValueError('Error! the number of points used for the periodicization exceeds %d\n'%(nn//2,))
        
        rs = raised_cosine(npoints)
        rs = np.concatenate([rs, np.ones(nn-2*npoints), np.flip(rs)])
    mean = np.mean(sig)
    return (sig - mean) * rs + mean



def raised_cosine(n, endpoint = False):
    """
    returns an array of size n containing a raised cosine from 0 to 1.
    If endpoint == True, then out[n] = 1,
    """
    x=np.arange(float(n))/(n-1) if endpoint else np.arange(float(n))/n
    x = x*np.pi
    return (-np.cos(x) +1)/2


def extend_signal(sig,npad, mode = 'asymw-periodic',**kwargs):

    """
    wrapper for numpy.pad

    provides aliases for some modes:

    Parameters
    ----------

    mode : str (default is 'constant')
        method of extension/padding.
        the modes are the same of numpy.pad.
        additional modes provided by this wrapper are

        'asymw' : correspond to mode = 'reflect', reflect_type='odd'
        'symw'  : correspond to mode = 'reflect', reflect_type='even'
        'asymw-periodic': correspond to 'asymw', but the padded region is multiplied by 
            a raised cosine (plus the mean) to make the signal periodic.
        example:
        >>>a=np.arange(10)
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>>wextend(a,4,mode = 'symw')
        array([4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5])
        >>>wextend(a,4,mode = 'asymw')
        array([-4, -3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,13])
    """

    if mode.lower() == 'symw':
        
        return np.pad(sig, npad, mode = 'reflect', reflect_type='even',**kwargs)
    
    if mode.lower() == 'asymw':
        
        return np.pad(sig, npad, mode = 'reflect', reflect_type='odd',**kwargs)

    if mode == 'asymw-periodic':
        
        shape = np.shape(sig)
        new_sig = np.pad(sig, npad, mode = 'reflect', reflect_type='odd',**kwargs) 
        newshape=new_sig.shape
        if len(shape) == 1:
            mean = np.mean(new_sig)
        
            rs = raised_cosine(npad,endpoint=True)
        
            new_sig[0:npad] = (new_sig[0:npad] - mean)*rs +mean
            new_sig[-npad:] = (new_sig[-npad:] - mean)*np.flip(rs) + mean
       
        else:
            new_sig=new_sig.reshape( (np.prod(newshape[0:-1]),newshape[-1]) )
            
            mean = np.mean(new_sig,axis=-1) 
            
            rsl = raised_cosine(npad[-1][0])
            rsr = np.flip(raised_cosine(npad[-1][-1]))
            
            for i in range(new_sig.shape[0]): 
                new_sig[i,0:npad[-1][0]] = (new_sig[i,0:npad[-1][0]] - mean[i])*rsl +mean[i]
                new_sig[i,-npad[-1][-1]:] = (new_sig[i,-npad[-1][-1]:] - mean[i])*rsr + mean[i]

            new_sig.reshape(newshape)

       
        return new_sig
    
    return np.pad(sig, npad, mode = mode,**kwargs)



#******************************************************************************
#*******************ARRAYS MANIPULATION TOOLS**********************************
#******************************************************************************



def wextend(sig, npad, mode = 'constant', **kwargs):
    """
    wrapper for numpy.pad

    provides aliases for some modes:

    Parameters
    ----------

    mode : str (default is 'constant')
        method of extension/padding.
        the modes are the same of numpy.pad.
        additional modes provided by this wrapper are

        'asymw' : correspond to mode = 'reflect', reflect_type='odd'
        'symw'  : correspond to mode = 'reflect', reflect_type='even'

        example:
        >>>a=np.arange(10)
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>>wextend(a,4,mode = 'symw')
        array([4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5])
        >>>wextend(a,4,mode = 'asymw')
        array([-4, -3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,13])
    """

    if mode.lower() == 'symw':
        return np.pad(sig, npad, mode = 'reflect', reflect_type='even',**kwargs)
    if mode.lower() == 'asymw':
        return np.pad(sig, npad, mode = 'reflect', reflect_type='odd',**kwargs)

    return np.pad(sig, npad, mode = mode,**kwargs)



















