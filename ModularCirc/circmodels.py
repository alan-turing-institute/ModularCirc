import numpy as np
import sympy as sp
from .timeclass import TimeSeries


def resistor_model_flow(p_in:float, p_out:float, r:float) -> float:
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

def grounded_capacitor_model_pressure(v:float, v_ref:float, c:float) -> float:
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

def chamber_volume_rate_change(q_in:float, q_out:float) -> float:
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
    
def get_softplus_max(alpha:float) -> function:
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

def non_ideal_diode_flow(p_in:float, p_out:float, r:float, max_func:function=relu_max) -> float:
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
    return max_func([p_in - p_out, 0.0]) / r


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
        return 0.5 * np.exp(-(t - t_tr)/tau)
    
def chamber_pressure_function(v:float, t:float, activation_function, active_law, passive_law) ->float:
    """
    Generic function returning the chamber pressure at a given time for a given imput

    Args:
        v (float): current volume
        t (float): current time
        activation_function (procedure): activation function
        active_law (procedure): active p-v relation
        passive_law (procedure): passive p-v relation

    Returns:
        float: pressure
    """
    a = activation_function(t)
    return a * active_law(v,t) + (1 - a) * passive_law(v,t)

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
    

class RLSeriesModel():
    def __init__(self, time_series : TimeSeries) -> None:
        self.time_series = time_series
        
        p_in, p_out, q = sp.symbols('p_in, p_out, q')
        dqdt = sp.symbols('dqdt')
        dt   = sp.symbols('dt')
        
        L, R = sp.symbols('L, R')
        
        formula = (p_in - p_out) / dt - q - L / R * dqdt
        
        print(formula)