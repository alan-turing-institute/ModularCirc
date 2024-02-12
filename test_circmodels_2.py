from ModularCirc.HelperRoutines import activation_function_1, time_shift
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 800.0, 1.0)

t_max = 150.
t_tr  = 1.5 * t_max
tau   = 25.

def time_shift(t, shift, T):
    if t <= T - shift:
        return t + shift
    else:
        return t + shift - T

laf_la = np.vectorize(lambda t, t_max=t_max, t_tr=t_tr, tau=tau : activation_function_1(time_shift(t, 100., 800.), 
                                                                               t_max=t_max, t_tr=t_tr, tau=tau))

t_max = 280.
t_tr  = 1.5 * t_max
tau   = 25.

laf_lv = np.vectorize(lambda t : activation_function_1(t, t_max=t_max, t_tr=t_tr, tau=tau))


plt.plot(t, laf_la(t), label='la')
plt.plot(t, laf_lv(t), label='lv')
plt.show()

