
import numpy as np
import IF_v8_3e as itf

n=1024
size=3
sigma=3

from scipy.io import loadmat
from blombly.tools import time
tt=time.timeit()
MM = loadmat('prefixed_double_filter.mat')['MM'].flatten()
a = itf.get_mask_v1_1(MM, size,True,1e-12)

x = np.linspace(0,2*np.pi,n,endpoint=False)
y = np.sin(2*x) + np.cos(10*x+2.3)


y = np.zeros(n)

y[n//2] = 1

opts = itf.Settings(verbose=True)
tt.tic; imf1n,insteps,SD = itf.compute_imf_numba(y,a,opts); tt.toc
tt.tic; imf1f,insteps,SD = itf.compute_imf_fft(y,a,opts); tt.toc

tt.tic; imf2n,insteps,SD = itf.compute_imf_numba(y-imf1n,a,opts); tt.toc
tt.tic; imf2f,insteps,SD = itf.compute_imf_fft(y-imf1f,a,opts); tt.toc
tt.tic; imf3n,insteps,SD = itf.compute_imf_numba(y-imf1n-imf2n,a,opts); tt.toc
tt.tic; imf3f,insteps,SD = itf.compute_imf_fft(y-imf1f-imf2f,a,opts); tt.toc
tt.tic; imf4f,insteps,SD = itf.compute_imf_fft(y-imf1f-imf2f-imf3f,a,opts); tt.toc

import pylab as plt
plt.ion()
fig,ax = plt.subplots(3,1)
ax[0].plot(y)
ax[0].plot(y-imf1f,'-c')
ax[0].plot(y-imf1f-imf2f,'--g')
ax[0].plot(y-imf1f-imf2f-imf3f,'-.b')
ax[1].plot(imf1f,'-c')
ax[1].plot(imf2f,'--g')
ax[1].plot(imf3f,'-.b')
ax[2].plot(np.abs(np.fft.fft(y)),'b')
ax[2].plot(np.abs(np.fft.fft(y-imf1f)))
ax[2].plot(np.abs(np.fft.fft(y-imf1f-imf2f)))
ax[2].plot(np.abs(np.fft.fft(y-imf1f-imf2f-imf3f)))
ax[2].plot(np.abs(np.fft.fft(-imf1f-imf2f-imf3f)))
