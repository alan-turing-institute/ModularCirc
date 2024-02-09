import numpy as np  
from .Time import TimeClass

def resistor_model_flow(t:float, p_in:float, p_out:float, r:float) -> float:
    """
    Resistor model.

    Args:
        p_in (float): input pressure
        p_out (float): ouput pressure
        r (float): resistor constant

    Returns:
        float: q (flow rate through resistive unit)
    """
    return (p_in - p_out) / r

def resistor_model_dp(q_in:float, r:float) -> float:
    return q_in * r

def grounded_capacitor_model_pressure(t:float, v:float, v_ref:float, c:float) -> float:
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
    return (v - v_ref) / c

def grounded_capacitor_model_dpdt(t:float, q_in:float, q_out:float, c:float) -> float:
    return (q_in - q_out) / c

def chamber_volume_rate_change(t:float, q_in:float, q_out:float) -> float:
    """
    Volume change rate in chamber

    Args:
        q_in (float): _description_
        q_out (float): _description_

    Returns:
        float: _description_
    """
    return q_in - q_out

def relu_max(val:float) -> float: 
    return np.max([val, 0.0])

def softplus(val:float, alpha:float) -> float:
    if alpha * val <= 20.0:
        return 1/ alpha * np.log(1 + np.exp(alpha * val))
    else:
        return val
    
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

def non_ideal_diode_flow(p_in:float, p_out:float, r:float, max_func=relu_max) -> float:
    """
    Nonideal diode model with the option to choose the re

    Args:
    -----
        p_in (float): input pressure
        p_out (float): output pressure
        r (float): valve constant resistance
        max_func (function): function that dictates when valve oppens

    Returns:
        float: q (flow rate through valve)
    """
    # print('*', max_func(p_in - p_out) / r)
    return max_func(p_in - p_out) / r


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
    if p_in > p_out:
        return (p_in - p_out) / r_o
    else:
        return (p_in - p_out) / r_r
    
def activation_function_1(t:float, t_max:float, t_tr:float, tau:float) -> float:
    """
    Activation function that dictates the transition between the passive and active behaviors.
    Based on the definition used in Naghavi et al (2024).

    Args:
        t (float):     current time within the cardiac cycle
        t_max (float): time to peak tension
        t_tr (float):  transition time
        tau (float):   the relaxation time constant

    Returns:
        float: activation function value
    """
    if t <= t_tr:
        return 0.5 * (1.0 - np.cos(np.pi * t / t_max))
    else:
        coeff = 0.5 * (1.0 - np.cos(np.pi * t_tr / t_max))
        return  np.exp(-(t - t_tr)/tau) * coeff
    
def activation_function_2(t:float, t_max:float, t_tr:float, tau:float) -> float:
    if t < t_max:
        return 0.5 * (1.0 - np.cos(np.pi * t / t_max))
    elif t - t_max < tau:
        return 0.5 * (1.0 + np.cos(np.pi * (t - t_max) / tau))
    else:
        return 0.0

def chamber_linear_elastic_law(v:float, E:float, v_ref:float, *args, **kwargs) -> float:
    """
    Linear elastance model

    Args:
        v (float): volume
        E (float): Elastance
        v_ref (float): reference volume
        
    Returns:
        float: chamber pressure
    """
    return E * (v - v_ref)

def chamber_exponential_law(v:float, E:float, k:float, v_ref:float, *args, **kwargs) -> float:
    """
    Exponential chamber law

    Args:
        v (float): volume
        E (float): elastance constant
        k (float): exponential factor
        v_ref (float): reference volume

    Returns:
        float: chamber pressure
    """
    return E * np.exp(k * (v - v_ref) - 1)

def chamber_pressure_function(t:float, v:float, v_ref:float, E_pas:float, E_act:float,
                              activation_function = activation_function_1, 
                              active_law = chamber_linear_elastic_law, 
                              passive_law = chamber_linear_elastic_law,
                              *args, **kwargs) ->float:
    """
    Generic function returning the chamber pressure at a given time for a given imput

    Args:
    -----
        t (float): current time
        v (float): current volume
        v_ref (float) : reference volume
        activation_function (procedure): activation function
        active_law (procedure): active p-v relation
        passive_law (procedure): passive p-v relation

    Returns:
    --------
        float: pressure
    """
    a = activation_function(t)
    return (a * active_law(v=v, v_ref=v_ref,t=t, E=E_act, **kwargs) 
            + (1 - a) * passive_law(v=v, v_ref=v_ref, t=t, E=E_pas, **kwargs))
    
def time_shift(t:float, shift:float, time_obj:TimeClass):
    if t < time_obj.tcycle - shift:
        return t + shift
    else:
        return t + shift - time_obj.tcycle


BOLD = '\033[1m'
YELLOW = '\033[93m'
END  = '\033[0m'

def bold_text(str_:str):
    return BOLD + YELLOW + str_ + END