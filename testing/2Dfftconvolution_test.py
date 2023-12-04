# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:12:30 2023

@author: darmo
"""

# Create a 2D periodic field with some random values
import numpy as np
import MIF_v3e as mif

field = np.random.rand(64, 64) + 1j*np.random.rand(64,64)
kmax=4
field[:,kmax:-kmax-1] = 0
field[kmax:-kmax-1,:] = 0
field = np.fft.ifft2(field).real
# Create a Gaussian kernel

MM = mif.loadmat(mif.get_window_file_path())['MM'].flatten()
m=(9,5)

kernel = mif.get_mask_2D_v3(MM,m)
# Convolve the field with the kernel
#output = convolve_periodic_field(field, kernel)

field = np.zeros([64,64])
field[31,31]=1
kpad = np.pad(kernel,((0,field.shape[0]-kernel.shape[0]),(0,field.shape[1]-kernel.shape[1])))
kpad = np.roll(kpad,(-m[0],-m[1]),(0,1))
conv=np.fft.irfft2(np.fft.rfft2(field)*np.fft.rfft2(kpad)) 
#conv = np.roll(conv,(-32,-32),(0,1))
# Display the results
import pylab as plt
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(field, cmap='gray')
ax1.set_title('Original field')
ax2.imshow(conv, cmap='gray')
ax2.set_title('Convolved field')
plt.ion()
plt.show()

