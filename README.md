# FAST ITERATIVE FILTERING - python package

This package contains the python classes and methods for the Iterative Filtering (IF), Fast Iterative Filtering (FIF), Multidimensional Iterative Filtering (MIF), the Multivariate Iterative Filtering (MvFIF) and the Multivariate Fast Iterative Filtering (MvFIF) algorithms.

## Definitions ##
* IF: Iterative Filtering is an adaptive method for decomposing a 1D signal into a set of Intrinsic Mode Components (IMC) plus a trend. These components are simple oscillating functions whose average frequency is well behaved and form a complete and nearly orthogonal basis of the original signal. In this repository, IF is made fast by using FFT convolution (similar to FIF but without the problem of having a periodic signal).

* FIF: builds on iterative filtering and combines it with FFT to make it faster. It requires, however, periodic signals.

* MIF: it is used to decompose multidimensional signals (currently only 2D and only defined on periodic domains).
Other versions (e.g. MIF multidimensional 3D) if present, are currently experimental and should be used with caution.

* MvFIF: is the multivariate version of FIF, designed to decompose multichannel signals at once (e.g. components of a vector). It requires, however, periodic signals.

* MvIF: Same as MvFIF, but without the problem of having a periodic signal.

* IMFogram: the IMFogram method, is contained in fifpy.IMFogram_v1. This program is experimental and should be used with care.

### Notes ###
This repository is a complete rewriting of the original matlab codes by A. Cicone.


### Dependencies ###
The package has been written and tested in python3.8, should be compatible with python <3.13, due to requirement from numba.

Dependencies: scipy, numpy, numba, scikit-learn, attrdict, matplotlib

### Install ###

The package is available on PyPI. To install it simply type

```
pip install fifpy
```


### Examples ###

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

fifpy has been written and is maintained by Emanuele Papini - INAF (emanuele.papini@inaf.it).

The original code and algorithm conceptualization are authored by Antonio Cicone - University of L'Aquila (antonio.cicone@univaq.it).

Please feel free to contact us would you need any help in properly using fifpy and to report bug issues.

### Links ###
http://people.disim.univaq.it/~antonio.cicone/Software.html

Github: https://github.com/EmanuelePapini/fifpy

### References ###
1) A. Cicone, H. Zhou. [Numerical Analysis for Iterative Filtering with New Efficient Implementations Based on FFT.](https://arxiv.org/abs/1802.01359) Numerische Mathematik, 147 (1), pages 1-28, 2021. doi: 10.1007/s00211-020-01165-5

2) A. Cicone and E. Pellegrino. [Multivariate Fast Iterative Filtering for the decomposition of nonstationary signals.](https://arxiv.org/abs/1902.04860) IEEE Transactions on Signal Processing, Volume 70, pages 1521-1531, 2022. doi: 10.1109/TSP.2022.3157482

3) A. Cicone, W. S. Li, H. Zhou. [New theoretical insights in the decomposition and time-frequency representation of nonstationary signals: the IMFogram algorithm.](https://www.sciencedirect.com/science/article/abs/pii/S1063520324000113) Applied and Computational Harmonic Analysis, 71, 101634, 2024. doi: 10.1016/j.acha.2024.101634

4) A. Cicone, H. Zhou. [Multidimensional Iterative Filtering method for the decomposition of high-dimensional non-stationary signals.](https://doi.org/10.4208/nmtma.2017.s05) Cambridge Core in Numerical Mathematics: Theory, Methods and Applications, Volume 10, Issue 2, Pages 278-298, 2017. doi: 10.4208/nmtma.2017.s05 

5) E. Papini et al. [Multidimensional Iterative Filtering: a new approach for investigating plasma turbulence in numerical simulations.](https://doi.org/10.1017/S0022377820001221) Journal of Plasma Physics, Volume 86, Issue 5, 871860501, 2020. doi:10.1017/S0022377820001221

6) E. Papini et al. [Spacetime Hall-MHD Turbulence at Sub-ion Scales: Structures or Waves?](https://iopscience.iop.org/article/10.3847/2041-8213/ac11fd/pdf) The Astrophysical Journal Letters, 917:L12 (7pp), 2021. doi: 10.3847/2041-8213/ac11fd