import numpy as np
from . import MvIF 
import pylab as plt
from . import time

profile = False
if profile:
    import cProfile

timeit = time.timeit()
plt.ion()

import numpy as np
from . import MvIF
n=30000
size=11
sigma=3

x = np.linspace(0,6*np.pi,n,endpoint=False)
y2 = np.sin(2*x) 
y1 = np.cos(10*x+2.3)
y = np.sin(2*x) + np.cos(10*x+2.3)

z2 = np.cos(4*x)
z1 = np.sin(12*x+1.2)
z = z1 + z2

sig = np.vstack((y,z))
fif = MvIF(verbose=True,Xi=2.0,ExtPoints=9)
if profile:
    timeit.tic; cProfile.run("fif.run(y)",'IF_profiled'); timeit.toc
else:
    timeit.tic; fif.run(sig); timeit.toc
#y = np.zeros(n)
#
#y[n//2] = 1
#y[n//2+size//2] = 4
#y[n-3] = 1
#(convhigh,SD) = convolve_high(y,a)
IMF = fif.IMC

fig,ax = plt.subplots(IMF.shape[0]+1,2)
ax = ax.transpose()
ax[0,0].plot(y)
ax[1,0].plot(z)

for i in range(IMF.shape[0]):
    ax[0,i+1].plot(IMF[i,0])
    ax[1,i+1].plot(IMF[i,1])
#[plt.plot(i) for i in IMF]
ax[0,1].plot(y1,'--')
ax[0,2].plot(y2,'--')
ax[1,1].plot(z1,'--')
ax[1,2].plot(z2,'--')

