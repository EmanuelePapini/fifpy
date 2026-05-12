import csespy
import numpy as np
fpath = '/CSES_Data/CSES01/'
orbitn='028981'

css = csespy.CSES(fpath,orbitn=orbitn)


css.load_CSES('EFD_ELF')

df = css.data.EFD_ELF[2048:2048*100]

import fifpy

fif = fifpy.MvIF(imf_method='fft_numba',timeit=True,BCmode='wrap')
in_f = np.array([df[i].values for i in ['Ex','Ey','Ez']])
fif.run(in_f,preprocess='extend-periodic')

fif2 = fifpy.MvIF(imf_method='fft',timeit=True,BCmode='wrap')
fif2.run(in_f,preprocess='extend-periodic')

fif3 = fifpy.MvIF(imf_method='fft_adv',timeit=True,BCmode='wrap')
fif3.run(in_f,preprocess='extend-periodic')

fif4 = fifpy.MvIF(imf_method='fft_adv',timeit=True,BCmode='wrap',NumSteps=4)
fif4.run(in_f,preprocess='extend-periodic')