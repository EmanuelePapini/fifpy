# FAST ITERATIVE FILTERING python package

This repository contains the python for the Iterative Filtering (IF), Fast Iterative Filtering (FIF), Multidimensional Iterative Filtering (MIF), the Multivariate Iterative Filtering (MvFIF) and the Multivariate Fast Iterative Filtering (MvFIF) algorithms.

## Definitions ##
* IF: Iterative Filtering is an adaptive method for decomposing a 1D signal into a set of Intrinsic Mode Components (IMC) plus a trend. These components are simple oscillating functions whose average frequency is well behaved and form a complete and nearly orthogonal basis of the original signal. In this repository, IF is made fast by using FFT convolution (similar to FIF but without the problem of having a periodic signal).

* FIF: builds on iterative filtering and combines it with FFT to make it faster. It requires, however, periodic signals.

* MIF: it is used to decompose multidimensional signals (Currently only 2D and only defined on periodic domains).
Other versions (e.g. MIF multidimensional 3D) if present, are currently experimental and should be used with caution.

* MvFIF: is the multivariate version of FIF, designed to decompose multichannel signals at once (e.g. components of a vector). It requires, however, periodic signals.

* MvIF: Same as MvFIF, but without the problem of having a periodic signal.

In the package, IMFogram_v1.py contains the methods to calculate the IMFogram (see https://ui.adsabs.harvard.edu/abs/2020arXiv201114209B/abstract)

### Notes ###
This repository is a complete rewriting of the original matlab code by A. Cicone.


### Dependencies ###
The package has been written and tested in python3.

Dependencies: scipy, numpy, numba, time, pyfftw (optional),  (plus other standard libraries that should be already installed)

### Install ###

Simply download the repository in the desired folder to start using it.
If you have a PYTHONPATH already set, you can put the fifpy folder directly there so that you can import the package from everywhere.

example: assuming fifpy is located in the PYTHONPATH or in the local path from where python3 is been executed 

```
#create the signal to be analyzed
import numpy as np
x = np.linspace(0,2*np.pi,100,endpoint=False)
y = np.sin(2*x) + np.cos(10*x+2.3)
        
#do the FIF analysis
import fifpy
    
fif=fifpy.IF()
fif.run(y)
#plot the results
import pylab as plt
plt.ion()
plt.figure()
plt.plot(x,y,label='signal')
[plt.plot(x,fif.data['IMC'][i,:],label = 'IMC#'+str(i)) for i in range(fif.data['IMC'].shape[0])]
plt.legend(loc='best')

```

### Contacts ###

fifpy has been written by Emanuele Papini - INAF (emanuele.papini@inaf.it).

The original code and algorithm conceptualization are authored by Antonio Cicone - University of L'Aquila (antonio.cicone@univaq.it).

Please feel free to contact us would you need any help in getting fifpy up and running.

### Links ###
 http://people.disim.univaq.it/~antonio.cicone/Software.html

### References ###
1) A. Cicone, H. Zhou. [Numerical Analysis for Iterative Filtering with New Efficient Implementations Based on FFT.](https://arxiv.org/abs/1802.01359) Numerische Mathematik, 147 (1), pages 1-28, 2021. doi: 10.1007/s00211-020-01165-5
2) A. Cicone and E. Pellegrino. [Multivariate Fast Iterative Filtering for the decomposition of nonstationary signals.](https://arxiv.org/abs/1902.04860) IEEE Transactions on Signal Processing, Volume 70, pages 1521-1531, 2022. doi: 10.1109/TSP.2022.3157482


