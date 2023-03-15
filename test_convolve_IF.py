
import numpy as np
from numba import njit,prange
import if_parallel


def convolve_high(f,a):
        
    h = np.array(f)
    h_ave = np.zeros(len(h))
    ker = np.array(a)
    @njit
    def iterate_numba(h,h_ave,kernel):
        
        ker_size = len(kernel)
        hker_size = ker_size//2
        
        Nh = len(h)
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
        
        return h,SD
    
    return iterate_numba(h,h_ave,ker)

#def test_convolve_high_fft(f,a):
#
#    nf = f.size; na = a.size
#    if na < nf : 
#        npad = (nf-na)//2
#        npadr = npad+1 if nf //2 else npad
#        fa = np.fft.fft(np.pad(a,(0,nf-na)))
#    ff = np.fft.fft(f)
#
#    return f - np.fft.ifft((ff*fa.conj() + ff.conj()*fa)/2).real
#
#def fft_convolve1d(x,y): #1d cross correlation, fft
#
#    """ 1D convolution, using FFT """
#
#    yy=np.pad(y,(0,x.size-y.size))
#    fr=np.fft.fft(x)
#
#    fr2=np.fft.fft(np.flipud(yy))
#
#    cc=np.real(np.fft.ifft(fr*fr2))
#
#    return x-np.fft.fftshift(cc)
    
def fftconvolve1d(in1, in2, mode="same"):
    from scipy.signal.signaltools import _init_freq_conv_axes, _centered
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])

    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes,
                                          sorted_axes=False)

    s1 = in1.shape
    s2 = in2.shape

    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
             for i in range(in1.ndim)]

    ret = _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=True)

    return _centered(ret, s1).copy()

def _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=False):
    """Convolve two arrays in the frequency domain.

    This function implements only base the FFT-related operations.
    Specifically, it converts the signals to the frequency domain, multiplies
    them, then converts them back to the time domain.  Calculations of axes,
    shapes, convolution mode, etc. are implemented in higher level-functions,
    such as `fftconvolve` and `oaconvolve`.  Those functions should be used
    instead of this one.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    axes : array_like of ints
        Axes over which to compute the FFTs.
    shape : array_like of ints
        The sizes of the FFTs.
    calc_fast_len : bool, optional
        If `True`, set each value of `shape` to the next fast FFT length.
        Default is `False`, use `axes` as-is.

    Returns
    -------
    out : array
        An N-dimensional array containing the discrete linear convolution of
        `in1` with `in2`.

    """
    from scipy import fft as sp_fft
    if not len(axes):
        return in1 * in2

    complex_result = (in1.dtype.kind == 'c' or in2.dtype.kind == 'c')

    if calc_fast_len:
        # Speed up FFT by padding to optimal size.
        fshape = [
            sp_fft.next_fast_len(shape[a], not complex_result) for a in axes]
    else:
        fshape = shape

    if not complex_result:
        fft, ifft = sp_fft.rfftn, sp_fft.irfftn
    else:
        fft, ifft = sp_fft.fftn, sp_fft.ifftn

    sp1 = fft(in1, fshape, axes=axes)
    sp2 = fft(in2, fshape, axes=axes)

    ret = ifft(sp1 * sp2, fshape, axes=axes)

    if calc_fast_len:
        fslice = tuple([slice(sz) for sz in shape])
        ret = ret[fslice]

    return ret

def scipy_fftconvolve_high(f,a):
    from scipy.signal import fftconvolve

    return f - fftconvolve(f,a,mode='same')

def get_mask_v1_1(y, k,verbose,tol):
    """
    Rescale the mask y so that its length becomes 2*k+1.
    k could be an integer or not an integer.
    y is the area under the curve for each bar
    
    wrapped from FIF_v2_13.m
    
    """
    n = np.size(y)
    m = (n-1)//2
    k = int(k)

    if k<=m:

        if np.mod(k,1) == 0:
            
            a = np.zeros(2*k+1)
            
            for i in range(1, 2*k+2):
                s = (i-1)*(2*m+1)/(2*k+1)+1
                t = i*(2*m+1)/(2*k+1)

                s2 = np.ceil(s) - s

                t1 = t - np.floor(t)

                if np.floor(t)<1:
                    print('Ops')

                a[i-1] = np.sum(y[int(np.ceil(s))-1:int(np.floor(t))]) +\
                         s2*y[int(np.ceil(s))-1] + t1*y[int(np.floor(t))-1]
        else:
            new_k = int(np.floor(k))
            extra = k - new_k
            c = (2*m+1)/(2*new_k+1+2*extra)

            a = np.zeros(2*new_k+3)

            t = extra*c + 1
            t1 = t - np.floor(t)

            if k<0:
                print('Ops')
                a = []
                return a

            a[0] = np.sum(y[:int(np.floor(t))]) + t1*y[int(np.floor(t))-1]

            for i in range(2, 2*new_k+3):
                s = extra*c + (i-2)*c+1
                t = extra*c + (i-1)*c
                s2 = np.ceil(s) - s
                t1 = t - np.floor(t)

                a[i-1] = np.sum(y[int(np.ceil(s))-1:int(np.floor(t))]) +\
                         s2*y[int(np.ceil(s))-1] + t1*y[int(np.floor(t))-1]
            t2 = np.ceil(t) - t

            a[2*new_k+2] = np.sum(y[int(np.ceil(t))-1:n]) + t2*y[int(np.ceil(t))-1]

    else: # We need a filter with more points than MM, we use interpolation
        dx = 0.01
        # we assume that MM has a dx = 0.01, if m = 6200 it correspond to a
        # filter of length 62*2 in the physical space
        f = y/dx
        dy = m*dx/k
        # b = np.interp(list(range(1,int(m+1),m/k)), list(range(0,int(m+1))), f[m:2*m+1])
        b = np.interp(np.linspace(0,m,int(np.ceil(k+1))), np.linspace(0,m,m+1), f[m:2*m+1])

        a = np.concatenate((np.flipud(b[1:]), b))*dy

        if abs(LA.norm(a,1)-1)>tol:
            if verbose:
                print('\n\n Warning!\n\n')
                print(' Area under the mask equals %2.20f\n'%(LA.norm(a,1),))
                print(' it should be equal to 1\n We rescale it using its norm 1\n\n')
            a = a/LA.norm(a,1)
        
    return a

n=1000
size=200
sigma=3

from scipy.io import loadmat
from blombly.tools import time
tt=time.timeit()
MM = loadmat('prefixed_double_filter.mat')['MM'].flatten()
a = get_mask_v1_1(MM, size,True,1e-12)

x = np.linspace(0,2*np.pi,n,endpoint=False)
y = np.sin(2*x) + np.cos(10*x+2.3)


y = np.zeros(n)

y[n//2] = 1
y[n//2+size//2] = 4
y[n-1] = 1
tt.tic
(convhigh,SD) = convolve_high(y,a)
tt.toc

conv =if_parallel.iterfilt.convolve

tt.tic
fout = conv(y,a,y.size,a.size,4)
tt.toc

import pylab as plt
plt.ion()
plt.figure()
plt.plot(y)
plt.plot(convhigh)
plt.plot(fout)
tt.tic; fftf = scipy_fftconvolve_high(y,a);tt.toc
plt.plot(fftf,'--')
