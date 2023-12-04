import numpy as np
from . import MIF
import pylab as plt
from . import time

profile = False
if profile:
    import cProfile

timeit = time.timeit()
plt.ion()

n=512
size=11
sigma=3

x = np.linspace(0,2*np.pi,n,endpoint=False)
y2 = np.sin(2*x[:,None])*np.cos(2*x[None,:]+0.2) 
y1 = np.cos(10*x[:,None]+2.3)*np.sin(11*x[None,:])
trend = np.cos(x[:,None]+2.3)*np.sin(x[None,:]+1.3)
y = y1+y2+trend

fif = MIF(verbose=True,Xi=1.6,ExtPoints=3)
if profile:
    timeit.tic; cProfile.run("fif.run(y)",'IF_profiled'); timeit.toc
else:
    timeit.tic; fif.run(y); timeit.toc
#y = np.zeros(n)
#
#y[n//2] = 1
#y[n//2+size//2] = 4
#y[n-3] = 1
#(convhigh,SD) = convolve_high(y,a)
IMF = fif.IMC
from blombly import pylab as plt
plt.ion()
from blombly.pylab import plots as epp
fig,ax = plt.subplots(3,4)

epp.add_subplot_colorbar(fig,ax[0,0],ax[0,0].imshow(y))
ax[0,0].set_title('original signal')
epp.add_subplot_colorbar(fig,ax[1,0],ax[1,0].imshow(IMF.sum(axis=0)))
ax[1,0].set_title('IMF.sum')
epp.add_subplot_colorbar(fig,ax[2,0],ax[2,0].imshow(y-IMF.sum(axis=0)))
ax[2,0].set_title('Difference')
i = 1
epp.add_subplot_colorbar(fig,ax[0,i],ax[0,i].imshow(IMF[i-1]))
epp.add_subplot_colorbar(fig,ax[1,i],ax[1,i].imshow(y1))
epp.add_subplot_colorbar(fig,ax[2,i],ax[2,i].imshow(y1-IMF[i-1]))
i = 2
epp.add_subplot_colorbar(fig,ax[0,i],ax[0,i].imshow(IMF[i-1]))
epp.add_subplot_colorbar(fig,ax[1,i],ax[1,i].imshow(y2))
epp.add_subplot_colorbar(fig,ax[2,i],ax[2,i].imshow(y2-IMF[i-1]))
i = 3
epp.add_subplot_colorbar(fig,ax[0,i],ax[0,i].imshow(IMF[i-1]))
epp.add_subplot_colorbar(fig,ax[1,i],ax[1,i].imshow(trend))
epp.add_subplot_colorbar(fig,ax[2,i],ax[2,i].imshow(trend-IMF[i-1]))

#ax[0].plot(y)
#for i in range(IMF.shape[0]):
#    ax[i+1].plot(IMF[i])
#[plt.plot(i) for i in IMF]
#ax[1].plot(y1,'--')
#ax[2].plot(y2,'--')


