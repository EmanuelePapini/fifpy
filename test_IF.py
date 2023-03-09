import numpy as np
import IF_v8_3e as itf
import pylab as plt
plt.ion()

n=300
size=11
sigma=3

x = np.linspace(0,6*np.pi,n,endpoint=False)
y2 = np.sin(2*x) 
y1 = np.cos(10*x+2.3)
y = np.sin(2*x) + np.cos(10*x+2.3)

opts = itf.Settings(verbose=True,Xi=2.0)
IMF,stats = itf.IF_v8_3e(y,opts)
#y = np.zeros(n)
#
#y[n//2] = 1
#y[n//2+size//2] = 4
#y[n-3] = 1
#(convhigh,SD) = convolve_high(y,a)
fig,ax = plt.subplots(4)
ax[0].plot(y)
for i in range(IMF.shape[0]):
    ax[i+1].plot(IMF[i])
#[plt.plot(i) for i in IMF]
ax[1].plot(y1,'--')
ax[2].plot(y2,'--')


