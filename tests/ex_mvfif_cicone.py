#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from fifpy import MvFIF,MvIF


def generate_signals(t):
    c1 = (
        t / 2.0
        + (t / 30.0 + 3.0 / 5.0) * np.sin(2.0 * np.pi * t + np.pi / 2.0)
        + (2.0 - t / 30.0) * np.cos(2.0 * np.pi * np.power(t, 1.3) / 10.0)
    )

    c2 = (
        -t / 5.0 + (2.0 - t / 30.0) * np.sin(2.0 * np.pi * t) + 1.0 / 2.0 + t / 30.0
    ) * np.sin(6.0 * np.pi * (np.power(t, 1.5) / 20.0 + t))
    mv_signal = np.vstack([c1, c2])

    return mv_signal


def plot_signal(t, c1, c2):
    plt.plot(t, c1, label="c_1")
    plt.plot(t, c2, label="c_2")
    plt.show()


def plot_results(t, components, nr_to_plot=2):
    # Plot the first 'nr_to_plot' multivariate components
    plt.figure(figsize=(10, 20))

    for idx in range(nr_to_plot):
        plt.subplot(nr_to_plot, 1, idx + 1)
        plt.plot(t, components[idx, 0, :], label=f"Channel 1 — IMC {idx+1}")
        plt.plot(t, components[idx, 1, :], label=f"Channel 2 — IMC {idx+1}")
        plt.legend()
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Try to reproduce the results from section 4.2 of

    A. Cicone. 'Multivariate Fast Iterative Filtering for decomposition of nonstationary signals'
    ArXiv https://arxiv.org/abs/1902.04860
    """
    whichfif = 'MvIF'

    t = np.arange(0.0, 30.0, 0.01)
    mv_signal = generate_signals(t)

    # Run MvFIF
    mvfif = MvFIF(maxInner=200, ExtPoints=5, verbose=True,fft='numpy')
    #mvfif = MvIF(maxInner=200, ExtPoints=5, verbose=True,alpha=30)
    mvfif.run(mv_signal,preprocess='extend-periodic')

    # Extract IMCs: array of shape (nr_imcs, nr_channels, nr_samples)
    im_components = mvfif.IMC

    print(f"MvFIF found {len(im_components)} components")
    # plot original signal
    plot_signal(t, mv_signal[0], mv_signal[1])

    # plot imc components
    plot_results(t, im_components, nr_to_plot=10)
