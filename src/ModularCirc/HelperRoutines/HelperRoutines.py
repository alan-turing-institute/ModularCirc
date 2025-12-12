import numpy as np
from ..Time import TimeClass

import numba as nb

from collections.abc import Callable

@nb.njit(['float64(float64, float64[:], float64)'], cache=True)
def resistor_model_flow(t:float,
                        y:np.ndarray[float],
                        r:float
                        ) -> float:
    """
    Resistor model.

    Args:
        p_in (float): input pressure
        p_out (float): ouput pressure
        r (float): resistor constant

    Returns:
        float: q (flow rate through resistive unit)
    """
    p_in, p_out = y[:2]
    return (p_in - p_out) / r

@nb.njit(['float64(float64, float64[:], float64)'], cache=True)
def resistor_upstream_pressure(t:float,
                               y:np.ndarray[float],
                               r:float
                               )->float:
    q_in, p_out = y[:2]
    return p_out + r * q_in

@nb.njit(['float64(float64, float64)'], cache=True, inline='always')
def resistor_model_dp(q_in:float, r:float) -> float:
    return q_in * r

@nb.njit(['float64(float64, float64[:], float64, float64)'], cache=True)
def resistor_impedance_flux_rate(t:float,
                                 y:np.ndarray[float],
                                 r:float,
                                 l:float) -> float:
    """
    Resistor and impedance in series flux rate of change model.

    Args:
        t (float): current time
        p_in (float): inflow pressure
        p_out (float): outflow pressure
        q_out (float): outflow flux
        r (float): resistor constant
        l (float): impedance constant

    Returns:
        float: flux rate of change
    """
    p_in, p_out, q_out = y[:3]
    return (p_in - p_out - q_out * r ) / l

@nb.njit(['float64(float64, float64[:], float64, float64)'], cache=True)
def grounded_capacitor_model_pressure(t:float,
                                      y:np.ndarray[float],
                                      v_ref:float,
                                      c:float
                                      ) -> float:
    """
    Capacitor model with constant capacitance.

    Args:
    ----
        v (float): current volume
        v_ref (float): reference volume for which chamber pressure is zero
        c (float): capacitance constant

    Returns:
    --------
        float: pressure at input node
    """
    v = y[0]  # Extract scalar from array
    return (v - v_ref) / c

@nb.njit(['float64(float64, float64[:], float64, float64, float64)'], cache=True)
def grounded_nonlinear_capacitor_model_pressure(t: float,
                                                y: np.ndarray[float],
                                                v_ref: float,
                                                c0: float,
                                                p0: float
                                                ) -> float:
    v = y[0]
    return p0 + np.tan((v-v_ref) / c0)

@nb.njit(['float64(float64, float64[:], float64, float64)'], cache=True)
def grounded_capacitor_model_volume(t:float,
                                    y:np.ndarray[float],
                                    v_ref:float,
                                    c:float
                                    )->float:
    p = y[0]  # Extract scalar from array
    return v_ref + p * c

@nb.njit(['float64(float64, float64[:], float64, float64, float64)'], cache=True)
def grounded_nonlinear_capacitor_model_volume(t: float,
                                               y: np.ndarray[float],
                                               v_ref: float,
                                               c0: float,
                                               p0: float
                                               ) -> float:
    p = y[0]
    return v_ref + c0 * np.arctan(p - p0)

@nb.njit(['float64(float64, float64[:], float64)'], cache=True)
def grounded_capacitor_model_dpdt(t:float,
                                  y:np.ndarray[float],
                                  c:float
                                  ) -> float:
    q_in, q_out = y[:2]
    return (q_in - q_out) / c

@nb.njit(['float64(float64, float64[:], float64, float64)'], cache=True)
def grounded_nonlinear_capacitor_model_dpdt(t: float,
                                            y: np.ndarray[float],
                                            c0: float,
                                            p0: float
                                            ) -> float:
    q_in = y[0]
    q_out = y[1]
    p_in = y[2]
    return (q_in - q_out) * (1 + (p_in - p0)**2.) / c0
    

@nb.njit(['float64(float64, float64[:])'], cache=True)
def chamber_volume_rate_change(t:float,
                               y:np.ndarray[float]
                               ) -> float:
    """
    Volume change rate in chamber

    Args:
        q_in (float): _description_
        q_out (float): _description_

    Returns:
        float: _description_
    """
    q_in, q_out = y[:2]
    return q_in - q_out

@nb.njit(['float64[:](float64, float64[:,:])'], cache=True, parallel=True)
def chamber_volume_rate_change_vectorized(t:float, y_batch:np.ndarray[float]) -> np.ndarray[float]:
    """Vectorized version for batch processing multiple chambers simultaneously."""
    n_samples = y_batch.shape[0]
    result = np.empty(n_samples, dtype=np.float64)
    for i in nb.prange(n_samples):
        q_in, q_out = y_batch[i, :2]
        result[i] = q_in - q_out
    return result

@nb.njit(['float64(float64)'], cache=True, inline='always')
def relu_max(val:float) -> float:
    return np.maximum(val, 0.0)

@nb.njit(['float64(float64, float64)'], cache=True)
def softplus(val:float, alpha:float=0.2) -> float:
    """Softplus function used as a smooth rectifier (differentiable approximation to max(0, x))."""
    return np.log(1.0 + np.exp(alpha * val)) / alpha

def get_softplus_max(alpha:float):
    """
    Method for generating softmax lambda function based on predefined alpha values

    Args:
    ----
        alpha (float): softplus alpha value

    Returns:
    -------
        function: softplus function with fixed alpha
    """
    return lambda val : softplus(val=val, alpha=alpha)

@nb.njit(['float64(float64, float64[:], float64)'], cache=True)
def non_ideal_diode_flow(t:float,
                         y:np.ndarray[float],
                         r:float,
                         ) -> float:
    """
    Non-ideal diode model for resistive flow through a valve.

    Args:
    -----
        t (float): current time
        y (ndarray): state variables where y[0] is the pressure difference (dp)
        r (float): valve constant resistance

    Returns:
    -------
        float: q (flow rate through valve)
    """
    dp = y[0]
    return dp / r

@nb.njit(['float64(float64, float64[:], float64, float64)'], cache=True)
def simple_bernoulli_diode_flow(t:float,
                         y:np.ndarray[float],
                         CQ:float,
                         RRA:float=0.0
                         ) -> float:
    """
    Non-ideal diode model with the option to choose the re

    Args:
    -----
        p_in (float): input pressure
        p_out (float): output pressure
        r (float): valve constant resistance

    Returns:
        float: q (flow rate through valve)
    """
    p_in, p_out = y[:2]
    dp = p_in - p_out
    if dp >= 0.0:
        return CQ * np.sqrt(np.abs(dp))
    else:
        return -CQ * RRA * np.sqrt(np.abs(dp))

# @jit(cache=True, nopython=True)
def maynard_valve_flow(t:float,
                       y:np.ndarray[float],
                       CQ:float,
                       RRA:float=0.0
                       )->np.ndarray[float]:
    p_in, p_out, phi = y[:3]
    dp = p_in - p_out
    aeff = (1.0 - RRA) * phi + RRA
    return np.where(dp >= 0.0, aeff, -aeff) * CQ * np.sqrt(np.abs(dp))

@nb.njit(cache=True)
def maynard_phi_law(t:float,
                    y:nb.types.Array,
                    Ko:float,
                    Kc:float
                    )->nb.types.Array:
    p_in, p_out, phi = y[:3]
    dp = p_in - p_out
    return np.where(dp >= 0.0, Ko * (1.0 - phi) * dp, Kc * phi * dp)

@nb.njit(cache=True)
def maynard_impedance_dqdt(t:float,
                           y:nb.types.Array,
                           CQ:float,
                           R:float,
                           L:float,
                           RRA:float=0.0
                           )->nb.types.Array:
    p_in, p_out, q_in, phi = y[:4]
    dp = p_in - p_out
    aeff = (1.0 - RRA) * phi + RRA
    # Optimize division and power operations
    CQ_squared = CQ * CQ
    aeff_inv = 1.0 / aeff if aeff > 1.0e-5 else 0.0
    q_abs = np.abs(q_in)
    return np.where(aeff > 1.0e-5, 
                   (dp * aeff - q_in * R * aeff - q_abs * q_in * aeff_inv / CQ_squared) / L, 
                   0.0)

@nb.njit(['float64(float64, float64, float64, float64)'], cache=True)
def leaky_diode_flow(p_in:float, p_out:float, r_o:float, r_r:float) -> float:
    """
    Leaky diode model that outputs the flow rate through a leaky diode

    Args:
        p_in (float): input pressure
        p_out (float): output pressure
        r_o (float): outflow resistance
        r_r (float): regurgitant flow resistance

    Returns:
        float: q flow rate through diode
    """
    dp = p_in - p_out
    if dp >= 0.0:
        return dp/r_o
    else:
        return dp/r_r

@nb.njit(['float64(float64, float64, float64, float64, boolean)'], cache=True)
def activation_function_1(t:float, t_max:float, t_tr:float, tau:float, dt: bool=False) -> float:
    """
    Numba-optimized activation function that dictates the transition between 
    the passive and active behaviors. Based on the definition used in Naghavi et al (2024).

    Args:
        t (float):     current time within the cardiac cycle
        t_max (float): time to peak tension
        t_tr (float):  transition time
        tau (float):   the relaxation time constant
        dt (bool):     if True, return derivative

    Returns:
        float: activation function value or derivative
    """
    if not dt:
        if t <= t_tr:
            return 0.5 * (1.0 - np.cos(np.pi * t / t_max))
        else:
            coeff = 0.5 * (1.0 - np.cos(np.pi * t_tr / t_max))
            return np.exp(-(t - t_tr)/tau) * coeff
    else:
        if t <= t_tr:
            return 0.5 * np.pi / t_max * np.sin(np.pi * t / t_max)
        else:
            coeff = 0.5 * (1.0 - np.cos(np.pi * t_tr / t_max))
            return -np.exp(-(t - t_tr)/tau) * coeff / tau

@nb.njit(['float64(float64, float64, float64, boolean)'], cache=True)
def activation_function_2(t:float, tr:float, td:float, dt: bool=False) -> float:
    """
    Numba-optimized activation function with rise and decay phases.
    
    Args:
        t (float): current time
        tr (float): rise time
        td (float): decay time
        dt (bool): if True, return derivative
        
    Returns:
        float: activation function value or derivative
    """
    if not dt:
        if t < tr:
            return 0.5 * (1.0 - np.cos(np.pi * t / tr))
        elif t < td:
            return 0.5 * (1.0 + np.cos(np.pi * (t - tr) / (td - tr)))
        else:
            return 0.0
    else:
        if t < tr:
            return 0.5 * np.pi / tr * np.sin(np.pi * t / tr)
        elif t < td:
            return -0.5 * np.pi / (td - tr) * np.sin(np.pi * (t - tr) / (td - tr))
        else:
            return 0.0

@nb.njit(['float64(float64, float64, float64, boolean)'], cache=True)
def activation_function_3(t:float, tpwb:float, tpww:float, dt: bool=False) -> float:
    """
    Numba-optimized pulse wave activation function.
    
    Args:
        t (float): current time
        tpwb (float): pulse wave begin time
        tpww (float): pulse wave width
        dt (bool): if True, return derivative
        
    Returns:
        float: activation function value or derivative
    """
    if not dt:
        if t < tpwb:
            return 0.0
        elif t < tpwb + tpww:
            return 0.5 * (1.0 - np.cos(2.0 * np.pi * (t - tpwb) / tpww))
        else:
            return 0.0
    else:
        if t < tpwb:
            return 0.0
        elif t < tpwb + tpww:
            return np.pi / tpww * np.sin(2.0 * np.pi * (t - tpwb) / tpww)
        else:
            return 0.0




@nb.njit(['float64(float64, float64[:], float64, float64)'], cache=True)
def active_pressure_law(t:float, y:np.ndarray[float], E_act:float, v_ref:float) -> float:
    """
    Active pressure law for heart chambers (linear elastance).
    
    Args:
        t (float): current time
        y (ndarray): state variables [volume, ...]
        E_act (float): active elastance
        v_ref (float): reference volume
    
    Returns:
        float: active pressure
    """
    v = y[0]
    return E_act * (v - v_ref)

@nb.njit(['float64(float64, float64[:], float64, float64, float64)'], cache=True)
def passive_pressure_law(t:float, y:np.ndarray[float], E_pas:float, k_pas:float, v_ref:float) -> float:
    """
    Passive pressure law for heart chambers (exponential elastance).
    
    Args:
        t (float): current time
        y (ndarray): state variables [volume, ...]
        E_pas (float): passive elastance
        k_pas (float): exponential factor
        v_ref (float): reference volume
    
    Returns:
        float: passive pressure
    """
    v = y[0]
    return E_pas * (np.exp(k_pas * (v - v_ref)) - 1.0)



@nb.njit(['float64(float64, float64[:], float64)'], cache=True)
def active_dpdt_law(t:float, y:np.ndarray[float], E_act:float) -> float:
    """
    Active pressure derivative law.
    
    Args:
        t (float): current time
        y (ndarray): state variables [volume, q_in, q_out, ...]
        E_act (float): active elastance
    
    Returns:
        float: active pressure derivative
    """
    q_in, q_out = y[1], y[2]
    return E_act * (q_in - q_out)

@nb.njit(['float64(float64, float64[:], float64, float64, float64)'], cache=True)
def passive_dpdt_law(t:float, y:np.ndarray[float], E_pas:float, k_pas:float, v_ref:float) -> float:
    """
    Passive pressure derivative law.
    
    Args:
        t (float): current time
        y (ndarray): state variables [volume, q_in, q_out, ...]
        E_pas (float): passive elastance
        k_pas (float): exponential factor
        v_ref (float): reference volume
    
    Returns:
        float: passive pressure derivative
    """
    v, q_in, q_out = y[0], y[1], y[2]
    return E_pas * k_pas * np.exp(k_pas * (v - v_ref)) * (q_in - q_out)



@nb.njit(['float64(float64, float64[:], float64, float64, float64)'], cache=True)
def volume_from_pressure_nonlinear(t:float, y:np.ndarray[float], E_pas:float, v_ref:float, k_pas:float) -> float:
    """
    Calculate volume from pressure for nonlinear (exponential) elastance.
    
    Args:
        t (float): current time
        y (ndarray): state variables [pressure, ...]
        E_pas (float): passive elastance
        v_ref (float): reference volume
        k_pas (float): exponential factor
    
    Returns:
        float: volume
    """
    p = y[0]
    return v_ref + np.log(p / E_pas + 1.0) / k_pas



@nb.njit(['float64(float64, float64, float64)'], cache=True)
def time_shift(t:float, shift:float=np.nan, tcycle:float=0.0):
    if np.isnan(shift):
        return t
    elif t < tcycle - shift:
        return t + shift
    else:
        return t + shift - tcycle

@nb.njit(['void(float64[:], float64, float64, float64[:])'], cache=True, parallel=True)
def time_shift_inplace(t_array:np.ndarray[float], shift:float, tcycle:float, output:np.ndarray[float]):
    """In-place vectorized time shift to avoid memory allocation."""
    for i in nb.prange(len(t_array)):
        t = t_array[i]
        if np.isnan(shift):
            output[i] = t
        elif t < tcycle - shift:
            output[i] = t + shift
        else:
            output[i] = t + shift - tcycle
            
def compute_derivatives_batch(t: float, y: np.ndarray, funcs: list, out: np.ndarray):
    """
    Compute derivatives for a batch of functions.
    Fallback implementation for when Cython is not available.
    
    Args:
        t: time value
        y: state array (could be 1D or 2D if multiple indices)
        funcs: list of functions to call
        out: output array to store results
    """
    for i, func in enumerate(funcs):
        # If y is 2D, pass the i-th row; if 1D, pass the whole array
        if y.ndim == 2:
            out[i] = func(t, y[i])
        else:
            out[i] = func(t, y)

def compute_derivatives_batch_indexed(t: float, y: np.ndarray, ids: np.ndarray, funcs: list, out: np.ndarray):
    """
    Compute derivatives for a batch of functions with indexing.
    Fallback implementation for when Cython is not available.
    """
    for i, (idx, func) in enumerate(zip(ids, funcs)):
        out[i] = func(t, y[idx])

class GenTimeShifter:
    """
    Time shifter class for delayed activation functions.
    Fallback implementation for when Cython is not available.
    """
    def __init__(self, shift: float, tcycle: float):
        self.shift = shift
        self.tcycle = tcycle
    
    def __call__(self, t: float, dt: bool = False) -> float:
        # Numba time_shift doesn't have a dt parameter, just returns shifted time
        # If dt=True is requested, we'd need to compute the derivative, but
        # for a simple time shift the derivative is just 1.0
        if dt:
            return 1.0  # Derivative of time shift is 1
        else:
            return time_shift(t, self.shift, self.tcycle)

def gen_total_dpdt_fixed(_af, E_act: float, v_ref: float, E_pas: float, k_pas: float):
    """
    Generate total pressure derivative function.
    This is imported from ComponentFactoriesOptimized, included here for compatibility.
    """
    def func(t, y):
        _af_t = _af(t, dt=False)
        _d_af_dt = _af(t, dt=True)
        
        active_p_val = active_pressure_law(t=0.0, y=y, E_act=E_act, v_ref=v_ref)
        passive_p_val = passive_pressure_law(t=0.0, y=y, E_pas=E_pas, k_pas=k_pas, v_ref=v_ref)
        
        active_dpdt_val = active_dpdt_law(t=0.0, y=y, E_act=E_act)
        passive_dpdt_val = passive_dpdt_law(t=0.0, y=y, E_pas=E_pas, k_pas=k_pas, v_ref=v_ref)
        
        return (_d_af_dt * (active_p_val - passive_p_val) +
               _af_t * active_dpdt_val +
               (1. - _af_t) * passive_dpdt_val)
    return func


BOLD = '\033[1m'
YELLOW = '\033[93m'
END  = '\033[0m'

def bold_text(str_:str):
    return BOLD + YELLOW + str_ + END
