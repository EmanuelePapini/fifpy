
from fifpy import MvFIF
from fifpy.arrays import extend_signal
from matplotlib import pylab as plt
import numpy as np



def generate_signal():
    """
    Generate a multivariate signal based on the equations provided in the original code.
    """
    # Time vector
    dt = 0.01
    t = np.arange(dt, 40 + dt, dt)

    # Generate components
    x1 = 0.5 * t
    x2 = (t / 30 + 0.6) * np.sin(2 * np.pi * t + np.pi / 2)
    x3 = (2 - t / 30) * np.cos(2 * np.pi * 0.1 * t ** 1.3)
    x = x1 + x2 + x3

    y1 = -0.2 * t
    y2 = (-t / 30 + 2) * np.sin(2 * np.pi * t)
    y3 = (0.5 + t / 30) * np.sin(2 * np.pi * 3 * (0.05 * t ** 1.5 + t))
    y = y1 + y2 + y3

    return np.vstack([x, y]), np.vstack([x2, x3, x1]), np.vstack([ y3, y2, y1])


def extend_signal_mvfif(sig,*args, mode =['symw-periodic','asymw-periodic'],**kwargs):
    """
    Extends the signal using 'symw-periodic' for the first signal and 
    'asymw-periodic' for the second signal by means of 
    fifpy.arrays.extend_signal to extend a signal with various modes.
    """

    nchan = sig.shape[0]
    if isinstance(mode, list):
        if len(mode) != nchan:
            raise ValueError(f"Number of modes ({len(mode)}) does not match number of channels ({nchan}).")
    elif isinstance(mode, str):
        mode = [mode.lower()] * nchan
    
    new_sig = [extend_signal(sig[i], *args, mode=mode[i], **kwargs) for i in range(nchan)]
    return np.vstack(new_sig)



if __name__ == "__main__":
    # Generate the signal
    sig,c1,c2 = generate_signal()
    nt = sig.shape[1]
    # Extend the signal
    sig_ext = extend_signal_mvfif(sig, nt, mode=['asymw-periodic', 'asymw-periodic'])

    #mvfif = MvFIF(verbose=True, alpha = 100, Xi = 1.6,fft='numpy')
    mvfif = MvFIF(verbose=True, alpha = 100, Xi = 1.6,fft='numpy',ExtPoints=3,MaskLengthType = 'amp',MaxlogM=18000)
    mvfif = MvFIF(verbose=True, alpha = 80, Xi = 2.0,fft='numpy',ExtPoints=3,MaskLengthType = 'amp',MaxlogM=18000)
    mvfif.run(sig_ext,wshrink=nt)
    mvfif.orthogonalize(threshold=0.1)
    # Plot the original and extended signals
    fig,ax=plt.subplots(4,sharex=True)
    ax[0].plot(sig[0].T)
    [axi.plot(i) for axi,i in zip(ax[1:],c1)]
    ax[-1].set_xlabel('Time')

    imc=mvfif.IMC
    imc.shape
    #ax[1].plot(imc[0,0,:],'--')
    ax[1].plot(imc[:2,0,:].sum(axis=0),'--')
    ax[2].plot(imc[2,0,:],'--')
    ax[3].plot(imc[3,0,:],'--')
    
    fig2,ax2=plt.subplots(4,sharex=True)
    ax2[0].plot(sig[1].T)
    [ax2i.plot(i) for ax2i,i in zip(ax2[1:],c2)]
    ax2[-1].set_xlabel('Time')

    imc=mvfif.IMC
    imc.shape
    ax2[1].plot(imc[0,1,:],'--')
    ax2[2].plot(imc[1,1,:],'--')
    ax2[3].plot(imc[2:,1,:].sum(axis=0),'--')

    #plt.subplot(2, 1, 2)
    #plt.plot(sig_ext.T)
    #plt.title('Extended Signal')
    #plt.xlabel('Time')
    #plt.ylabel('Amplitude')

    #plt.tight_layout()
    #plt.show()


