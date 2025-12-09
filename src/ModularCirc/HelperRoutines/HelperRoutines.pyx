# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, exp, log, cos, sin, fabs, isnan, M_PI, tan, atan
cimport cython
from libc.stdio cimport printf

cnp.import_array()

# Declare numpy array types for better performance
ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double resistor_model_flow(double t, double[::1] y, double r) nogil:
    """
    Resistor model.
    
    Args:
        t: current time
        y: state array where y[0]=p_in, y[1]=p_out
        r: resistor constant
    
    Returns:
        flow rate through resistive unit
    """
    cdef double p_in = y[0]
    cdef double p_out = y[1]
    return (p_in - p_out) / r

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double resistor_upstream_pressure(double t, double[::1] y, double r) nogil:
    """Calculate upstream pressure from flow and downstream pressure."""
    cdef double q_in = y[0]
    cdef double p_out = y[1]
    return p_out + r * q_in

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double resistor_model_dp(double q_in, double r) nogil:
    """Inline pressure drop calculation."""
    return q_in * r

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double resistor_impedance_flux_rate(double t, double[::1] y, double r, double l) nogil:
    """
    Resistor and impedance in series flux rate of change model.
    
    Args:
        t: current time
        y: [p_in, p_out, q_out]
        r: resistor constant
        l: impedance constant
    
    Returns:
        flux rate of change
    """
    cdef double p_in = y[0]
    cdef double p_out = y[1]
    cdef double q_out = y[2]
    return (p_in - p_out - q_out * r) / l

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double grounded_capacitor_model_pressure(double t, double[::1] y, double v_ref, double c) nogil:
    """
    Capacitor model with constant capacitance.
    
    Args:
        t: current time
        y: [volume]
        v_ref: reference volume for zero pressure
        c: capacitance constant
    
    Returns:
        pressure at input node
    """
    cdef double v = y[0]
    return (v - v_ref) / c

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double grounded_nonlinear_capacitor_model_pressure(double t, double[::1] y, double v_ref, double c0, double p0) nogil:
    """
    Nonlinear capacitor model.
    
    Args:
        t: current time
        y: [volume]
        v_ref: reference volume for zero pressure
        c: capacitance constant
        k: nonlinearity factor
    
    Returns:
        pressure at input node
    """
    cdef double v = y[0]
    return p0 + tan((v - v_ref) / c0)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double grounded_capacitor_model_volume(double t, double[::1] y, double v_ref, double c) nogil:
    """Calculate volume from pressure."""
    cdef double p = y[0]
    return v_ref + p * c

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double grounded_nonlinear_capacitor_model_volume(double t, double[::1] y, double v_ref, double c0, double p0) nogil:
    """Calculate volume from pressure (nonlinear capacitor)."""
    cdef double p = y[0]
    return v_ref + c0 * atan(p - p0)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double grounded_capacitor_model_dpdt(double t, double[::1] y, double c) nogil:
    """Capacitor pressure derivative."""
    cdef double q_in = y[0]
    cdef double q_out = y[1]
    return (q_in - q_out) / c

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double grounded_nonlinear_capacitor_model_dpdt(double t, double[::1] y, double c0, double p0) nogil:
    """Nonlinear capacitor pressure derivative."""
    cdef double q_in = y[0]
    cdef double q_out = y[1]
    cdef double p = y[2]
    return (q_in - q_out) * (1.0 + (p - p0) * (p - p0)) / c0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double chamber_volume_rate_change(double t, double[::1] y) nogil:
    """
    Volume change rate in chamber.
    
    Args:
        t: current time
        y: [q_in, q_out]
    
    Returns:
        volume rate of change
    """
    cdef double q_in = y[0]
    cdef double q_out = y[1]
    return q_in - q_out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double relu_max(double val) nogil:
    """ReLU activation: max(0, val)."""
    return val if val > 0.0 else 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double softplus(double val, double alpha=0.2) nogil:
    """Softplus function: smooth approximation to ReLU."""
    return log(1.0 + exp(alpha * val)) / alpha

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double non_ideal_diode_flow(double t, double[::1] y, double r) nogil:
    """
    Non-ideal diode model for resistive flow through a valve.
    
    Args:
        t: current time
        y: [dp] pressure difference
        r: valve constant resistance
    
    Returns:
        flow rate through valve
    """
    cdef double dp = y[0]
    return dp / r

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double simple_bernoulli_diode_flow(double t, double[::1] y, double CQ, double RRA=0.0) nogil:
    """
    Bernoulli diode flow model.
    
    Args:
        t: current time
        y: [p_in, p_out]
        CQ: flow coefficient
        RRA: regurgitant resistance ratio
    
    Returns:
        flow rate through valve
    """
    cdef double p_in = y[0]
    cdef double p_out = y[1]
    cdef double dp = p_in - p_out
    
    if dp >= 0.0:
        return CQ * sqrt(dp)
    else:
        return -CQ * RRA * sqrt(-dp)

cpdef double maynard_valve_flow(double t, double[::1] y, double CQ, double RRA=0.0):
    """Maynard valve flow model with phi state."""
    cdef double p_in = y[0]
    cdef double p_out = y[1]
    cdef double phi = y[2]
    cdef double dp = p_in - p_out
    cdef double aeff = (1.0 - RRA) * phi + RRA
    cdef double sign = 1.0 if dp >= 0.0 else -1.0
    
    return sign * aeff * CQ * sqrt(fabs(dp))

cpdef double maynard_phi_law(double t, double[::1] y, double Ko, double Kc):
    """Maynard phi evolution law."""
    cdef double p_in = y[0]
    cdef double p_out = y[1]
    cdef double phi = y[2]
    cdef double dp = p_in - p_out
    
    if dp >= 0.0:
        return Ko * (1.0 - phi) * dp
    else:
        return Kc * phi * dp

cpdef double maynard_impedance_dqdt(double t, double[::1] y, double CQ, double R, double L, double RRA=0.0):
    """Maynard impedance flow derivative."""
    cdef double p_in = y[0]
    cdef double p_out = y[1]
    cdef double q_in = y[2]
    cdef double phi = y[3]
    cdef double dp = p_in - p_out
    cdef double aeff = (1.0 - RRA) * phi + RRA
    cdef double CQ_squared = CQ * CQ
    cdef double aeff_inv, q_abs
    
    if aeff > 1.0e-5:
        aeff_inv = 1.0 / aeff
        q_abs = fabs(q_in)
        return (dp * aeff - q_in * R * aeff - q_abs * q_in * aeff_inv / CQ_squared) / L
    else:
        return 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double leaky_diode_flow(double p_in, double p_out, double r_o, double r_r) nogil:
    """
    Leaky diode model.
    
    Args:
        p_in: input pressure
        p_out: output pressure
        r_o: outflow resistance
        r_r: regurgitant flow resistance
    
    Returns:
        flow rate through diode
    """
    cdef double dp = p_in - p_out
    if dp >= 0.0:
        return dp / r_o
    else:
        return dp / r_r

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double activation_function_1(double t, double t_max, double t_tr, double tau, bint dt=False) nogil:
    """
    Activation function (Naghavi et al 2024 model).
    
    Args:
        t: current time within cardiac cycle
        t_max: time to peak tension
        t_tr: transition time
        tau: relaxation time constant
        dt: if True, return derivative
    
    Returns:
        activation value or derivative
    """
    cdef double coeff
    
    if not dt:
        if t <= t_tr:
            return 0.5 * (1.0 - cos(M_PI * t / t_max))
        else:
            coeff = 0.5 * (1.0 - cos(M_PI * t_tr / t_max))
            return exp(-(t - t_tr) / tau) * coeff
    else:
        if t <= t_tr:
            return 0.5 * M_PI / t_max * sin(M_PI * t / t_max)
        else:
            coeff = 0.5 * (1.0 - cos(M_PI * t_tr / t_max))
            return -exp(-(t - t_tr) / tau) * coeff / tau

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double activation_function_2(double t, double tr, double td, bint dt=False) nogil:
    """
    Activation function with rise and decay phases.
    
    Args:
        t: current time
        tr: rise time
        td: decay time
        dt: if True, return derivative
    
    Returns:
        activation value or derivative
    """
    if not dt:
        if t < tr:
            return 0.5 * (1.0 - cos(M_PI * t / tr))
        elif t < td:
            return 0.5 * (1.0 + cos(M_PI * (t - tr) / (td - tr)))
        else:
            return 0.0
    else:
        if t < tr:
            return 0.5 * M_PI / tr * sin(M_PI * t / tr)
        elif t < td:
            return -0.5 * M_PI / (td - tr) * sin(M_PI * (t - tr) / (td - tr))
        else:
            return 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double activation_function_3(double t, double tpwb, double tpww, bint dt=False) nogil:
    """
    Pulse wave activation function.
    
    Args:
        t: current time
        tpwb: pulse wave begin time
        tpww: pulse wave width
        dt: if True, return derivative
    
    Returns:
        activation value or derivative
    """
    if not dt:
        if t < tpwb:
            return 0.0
        elif t < tpwb + tpww:
            return 0.5 * (1.0 - cos(2.0 * M_PI * (t - tpwb) / tpww))
        else:
            return 0.0
    else:
        if t < tpwb:
            return 0.0
        elif t < tpwb + tpww:
            return M_PI / tpww * sin(2.0 * M_PI * (t - tpwb) / tpww)
        else:
            return 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double active_pressure_law(double t, double[::1] y, double E_act, double v_ref) nogil:
    """
    Active pressure law (linear elastance).
    
    Args:
        t: current time
        y: [volume, ...]
        E_act: active elastance
        v_ref: reference volume
    
    Returns:
        active pressure
    """
    cdef double v = y[0]
    return E_act * (v - v_ref)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double passive_pressure_law(double t, double[::1] y, double E_pas, double k_pas, double v_ref) nogil:
    """
    Passive pressure law (exponential elastance).
    
    Args:
        t: current time
        y: [volume, ...]
        E_pas: passive elastance
        k_pas: exponential factor
        v_ref: reference volume
    
    Returns:
        passive pressure
    """
    cdef double v = y[0]
    return E_pas * (exp(k_pas * (v - v_ref)) - 1.0)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double active_dpdt_law(double t, double[::1] y, double E_act) nogil:
    """
    Active pressure derivative law.
    
    Args:
        t: current time
        y: [volume, q_in, q_out, ...]
        E_act: active elastance
    
    Returns:
        active pressure derivative
    """
    cdef double q_in = y[1]
    cdef double q_out = y[2]
    return E_act * (q_in - q_out)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double passive_dpdt_law(double t, double[::1] y, double E_pas, double k_pas, double v_ref) nogil:
    """
    Passive pressure derivative law.
    
    Args:
        t: current time
        y: [volume, q_in, q_out, ...]
        E_pas: passive elastance
        k_pas: exponential factor
        v_ref: reference volume
    
    Returns:
        passive pressure derivative
    """
    cdef double v = y[0]
    cdef double q_in = y[1]
    cdef double q_out = y[2]
    return E_pas * k_pas * exp(k_pas * (v - v_ref)) * (q_in - q_out)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double volume_from_pressure_nonlinear(double t, double[::1] y, double E_pas, double v_ref, double k_pas) nogil:
    """
    Calculate volume from pressure (nonlinear/exponential elastance).
    
    Args:
        t: current time
        y: [pressure, ...]
        E_pas: passive elastance
        v_ref: reference volume
        k_pas: exponential factor
    
    Returns:
        volume
    """
    cdef double p = y[0]
    return v_ref + log(p / E_pas + 1.0) / k_pas

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double time_shift(double t, double shift=0.0, double tcycle=0.0) nogil:
    """
    Time shift function for periodic signals.
    
    Args:
        t: current time
        shift: time shift amount
        tcycle: cycle period (default: 0.0)
    
    Returns:
        shifted time
    """
    if fabs(shift) < 1e-12:
        return t
    elif t < tcycle - shift:
        return t + shift
    else:
        return t + shift - tcycle


cdef class GenTimeShifter:
    """
    Cythonized time shifter callable class.
    
    This replaces Python partial functions for time shifting,
    providing a fully compiled C implementation with nogil capability.
    """
    cdef double shift
    cdef double tcycle
    
    def __init__(self, double shift, double tcycle):
        """
        Initialize the time shifter.
        
        Args:
            shift: time shift amount
            tcycle: cycle period
        """
        self.shift = shift
        self.tcycle = tcycle
    
    def __call__(self, double t):
        """
        Apply time shift to input time.
        
        Args:
            t: current time
        
        Returns:
            shifted time
        """
        return time_shift(t, shift=self.shift, tcycle=self.tcycle)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void compute_derivatives_batch(double ht, double[:, :] all_inputs, 
                                      object funcs, double[::1] results) except *:
    """
    Cythonized batch computation of derivatives for primary state variables.
    
    Optimized version that assumes:
    - funcs is a numpy array of callable objects
    - Each function has signature: func(t=double, y=double[::1]) -> double
    
    This minimizes Python overhead by:
    1. Using typed memoryviews for array access
    2. Iterating at C speed through the functions array
    3. Direct assignment to results without intermediate Python objects
    
    Args:
        ht: current time in the heart cycle
        all_inputs: 2D array where each row contains inputs for one derivative function
        funcs: numpy array of derivative functions (each accepts t and y, returns double)
        results: 1D output array to store computed derivatives (modified in-place)
    
    Note:
        While the function calls are still Python objects (cannot be nogil),
        the iteration and array access are optimized at the C level.
    """
    cdef int i
    cdef Py_ssize_t n_funcs = all_inputs.shape[0]
    cdef object func
    cdef double result
    
    # Iterate through functions using C-level loop
    for i in range(n_funcs):
        # Get function from array (Python object access)
        func = funcs[i]
        # Call with known signature - pass memoryview slice directly
        # This avoids creating intermediate Python objects for the arguments
        result = func(t=ht, y=all_inputs[i])
        # Direct assignment to typed memoryview
        results[i] = result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void compute_derivatives_batch_indexed(double ht, double[::1] y_temp,
                                              long[:, ::1] ids,
                                              object funcs, double[::1] results) except *:
    """
    Highly optimized derivative computation that extracts inputs on-the-fly.
    
    This version avoids creating the intermediate all_inputs array by:
    - Taking the full state vector y_temp
    - Taking index array ids where each row specifies which elements to extract
    - Extracting values directly in the C loop
    
    Args:
        ht: current time in the heart cycle
        y_temp: full state vector
        ids: 2D array of indices, where each row specifies inputs for one function
        funcs: numpy array of derivative functions
        results: 1D output array to store computed derivatives (modified in-place)
    """
    cdef int i
    cdef int j
    cdef int valid_count
    cdef Py_ssize_t n_funcs = ids.shape[0]
    cdef Py_ssize_t n_inputs = ids.shape[1]
    cdef object func
    cdef double result
    cdef long idx
    
    # Pre-allocate a buffer for function inputs
    cdef double[::1] input_buffer = np.empty(n_inputs, dtype=np.float64)
    
    # Iterate through each function
    for i in range(n_funcs):
        # Extract inputs for this function
        valid_count = 0
        for j in range(n_inputs):
            idx = ids[i, j]
            if idx >= 0:  # -1 is used as padding, skip it
                input_buffer[valid_count] = y_temp[idx]
                valid_count += 1
            else:
                break  # Stop when we hit padding
        
        # Get function and call it with only the valid inputs
        func = funcs[i]
        result = func(t=ht, y=input_buffer[:valid_count])
        results[i] = result


# Helper function for softplus (kept for API compatibility)
def get_softplus_max(double alpha):
    """Return a lambda function with fixed alpha for softplus."""
    return lambda val: softplus(val, alpha)


@cython.boundscheck(False)
@cython.wraparound(False)
def gen_total_dpdt_fixed(_af, double E_act, double v_ref, double E_pas, double k_pas):
    """
    Generate a total dp/dt function with fixed parameters.
    
    Args:
        _af: activation function
        E_act: active elastance
        v_ref: reference volume
        E_pas: passive elastance
        k_pas: exponential factor
    
    Returns:
        function that computes total dp/dt
    """
    # Bind numeric parameters as Python floats for safe capture in the nested Python function
    # (Cython cannot capture C-level variables from an outer scope).
    E_act_f = float(E_act)
    v_ref_f = float(v_ref)
    E_pas_f = float(E_pas)
    k_pas_f = float(k_pas)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def total_dpdt(double t, double[::1] y, _af=_af,
                   E_act=E_act_f, v_ref=v_ref_f, E_pas=E_pas_f, k_pas=k_pas_f):
        """
        Total dp/dt combining active and passive components.
        
        Args:
            t: current time
            y: [volume, q_in, q_out, ...]
        
        Returns:
            total pressure derivative
        """
        cdef double af_t = _af(t, dt=False)
        cdef double af_dt = _af(t, dt=True)

        return (af_dt * (active_pressure_law(t, y, E_act, v_ref) - passive_pressure_law(t, y, E_pas, k_pas, v_ref)) +
                af_t * active_dpdt_law(t, y, E_act) +
                (1.0 - af_t) * passive_dpdt_law(t, y, E_pas, k_pas, v_ref))
    
    return total_dpdt

# Terminal formatting helpers
BOLD = '\033[1m'
YELLOW = '\033[93m'
END  = '\033[0m'

def bold_text(str_):
    """Format text as bold yellow."""
    return BOLD + YELLOW + str_ + END
