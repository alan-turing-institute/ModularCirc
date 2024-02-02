import numpy as np
import sympy as sp
# from .timeclass import TimeSeries


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

TEMPLATE_TIME_SETUP_DICT = {
    'name'    :  'generic',
    'ncycles' :  5,
    'tcycle'  :  1.0,
    'dt'      :  0.1
 }

class TimeClass():
    def __init__(self, time_setup_dict) -> None:
        self._time_setup_dict = time_setup_dict
        self._initialize_time_array()
        self.cti = 0 # current time step index
        
    @property
    def ncycles(self):
        if 'ncycles' in self._time_setup_dict.keys():
            return self._time_setup_dict['ncycles']
        else: 
            return None
    
    @property 
    def tcycle(self):
        if 'tcycle' in self._time_setup_dict.keys():
            return self._time_setup_dict['tcycle']
        else:
            return None
    
    @property
    def dt(self):
        if 'dt' in self._time_setup_dict.keys():
            return self._time_setup_dict['dt']
        else:
            return None
    
    def _initialize_time_array(self):
        # discretization of on heart beat, used as template
        self._cycle_t = np.linspace(
            start= 0.0,
            stop = self.tcycle,
            num  = int(self.tcycle / self.dt)+1,
            dtype= np.float64
            )
        
        # discretization of the entire simulation duration
        self._sym_t = np.array(
            [t+cycle*self.tcycle for cycle in range(self.ncycles) for t in self._cycle_t[:-1]]
        )
        
        # array of the current time within the heart cycle
        self._sym_t_norm = np.array(
            [t for _ in range(self.ncycles) for t in self._cycle_t[:-1]]
        )
        
        # the total number of time steps including initial time step
        self.n_t = len(self._sym_t)
        
    def new_time_step(self):
        self.cti += 1
        
        
class StateVariable():
    def __init__(self, name:str, timeobj:TimeClass) -> None:
        self._name  = name
        self._to   = timeobj
        self._u     = np.zeros((timeobj.n_t,))
        self._dudt = None
        
    def __repr__(self) -> str:
        return f" > variable name: {self._name}"
        
    def set_dudt(self, function)->None:
        self._dudt = function
        
    def set_name(self, name)->None:
        # print(f" -- name {name} replaces {÷self._name}")
        self._name = name
        
    @property
    def name(self):
        return self._name
        

class Chamber():
    def __init__(self,
                 name,
                 time_object:TimeClass, 
                 ) -> None:
        self._name     = name
        self._to      = time_object
        
        self._P_i = StateVariable(name=name+'_P_i', timeobj=time_object)
        self._Q_i = StateVariable(name=name+'_Q_i', timeobj=time_object)
        self._P_o = StateVariable(name=name+'_P_o', timeobj=time_object)
        self._Q_o = StateVariable(name=name+'_Q_o', timeobj=time_object)
        self._V   = StateVariable(name=name+'_V' , timeobj=time_object)
        
        self._V.set_dudt(function=chamber_volume_rate_change)
        return
        
    def __repr__(self) -> str:
        var = (f" Component {self._name}" + '\n' +
               f" - P_i " + str(self._P_i) + '\n' +
               f" - Q_i " + str(self._Q_i) + '\n' +
               f" - P_o " + str(self._P_o) + '\n' +
               f" - Q_o " + str(self._Q_o) + '\n'
               )
        return var
                
    @property
    def P_i(self):
        return self._P_i._u
    
    @property
    def P_o(self):
        return self._P_o._u
    
    @property
    def Q_i(self):
        return self._Q_i._u
    
    @property
    def Q_o(self):
        return self._Q_o._u
    
    @property
    def V(self):
        return self._V._u
    
    def comp_dvdt(self, intq:int=None, q_i:float=None, q_o:float=None):
        if intq is not None:
            return self.Q_i[intq] - self.Q_o[intq]
        elif q_i is not None and q_o is not None:
            return q_i - q_o
        
    def make_unique_io_state_variable(self, q_flag:bool=False, p_flag:bool=True) -> None:
        if q_flag: 
            self._Q_o = self._Q_i
            self._Q_o.set_name(self._name + '_Q')
        if p_flag: 
            self._P_o = self._P_i
            self._P_o.set_name(self._name + '_P')
        return 
            
class Rc_component(Chamber):
    def __init__(self, 
                 name:str, 
                 time_object:TimeClass, 
                #  main_var:str, 
                 r:float, 
                 c:float, 
                 v_ref:float
                 ) -> None:
        # super().__init__(time_object, main_var)
        super().__init__(time_object=time_object, name=name)
        self.R = r
        self.C = c
        self.V_ref = v_ref
        
    def comp_dpdt(self, intq:int=None, q_in:float=None, q_out:float=None):
        if intq is None and q_in is None and q_out is None:
            raise Exception("You have to either asign a value to intq or to q_in and q_out.")
        elif intq is None and q_in is None and q_out is not None:
            raise Exception("If you define dpdt in terms of q_in and q_out both have to be assigned.")
        elif intq is None and q_in is not None and q_out is None:
            raise Exception("If you define dpdt in terms of q_in and q_out both have to be assigned.")
        
        if intq is not None:    
            return 1.0 /  self.C * (self.Q_i[intq] - self.Q_o[intq])
        elif q_in is not None and q_out is not None:
            return 1.0 / self.C * (q_in - q_out)
        else: 
            raise Exception("Input case not covered.")
    
    def comp_p_i(self, intv:int):
        return (self.V[intv] - self.V_ref) / self.C
    
    def comp_q_o(self, intp:int=None, p_i:float=None, p_o:float=None):
        if intp is not None:
            return (self.P_i[intp] - self.P_o[intp]) / self.R
        elif p_i is not None and p_o is not None:
            return (p_i - p_o) / self.R
        else:
            raise Exception("Input case not covered")
    
    
class Valve_non_ideal(Chamber):
    def __init__(self, 
                 name:str,
                 time_object: TimeClass,
                 r:float, 
                 max_func
                 ) -> None:
        super().__init__(name=name, time_object=time_object)
        # allow for pressure gradient but not for flow
        self.make_unique_io_state_variable(q_flag=True, p_flag=False) 
        # setting the resistance value
        self.R = r
        self.max_func = max_func
        
    def comp_q(self, intp:int=None, p_in:float=None, p_out:float=None):
        if intp is not None:
            return self.max_func(self.P_i[intp] - self.P_o[intp]) / self.R
        elif p_in is not None and p_out is not None:
            return (p_in - p_out) / self.R
        
    
class HC_constant_elastance(Chamber):
    def __init__(self, 
                 name:str,
                 time_object: TimeClass,
                 E_pas: float, 
                 E_act: float,
                 V_ref: float,
                 activation_function_template = activation_function_1,
                 *args, **kwargs
                 ) -> None:
        super().__init__(name=name, time_object=time_object)
        self._af = lambda t : activation_function_template(t, *args, **kwargs)
        self.E_pas = E_pas
        self.E_act = E_act
        self.V_ref = V_ref
        self.eps = 1.0e-3
        
        self.make_unique_io_state_variable(p_flag=True, q_flag=False)
        
    def comp_E(self, t:float) -> float:
        return self._af(t) * self.E_act + (1.0 - self._af(t)) * self.E_pas
    
    def comp_dEdt(self, t:float) -> float:
        return (self.comp_E(t + self.eps) - self.comp_E(t - self.eps)) / 2.0 / self.eps
    
    def comp_p(self, intt:int, intv:int):
        e = self.comp_E(self._to._sym_t_norm[intt])
        v = self.V[intv]
        return e * (v - self.V_ref)
    
    def comp_dpdt(self, intt:int, intq:int=None, V:float=None, q_i:float=None, q_o:float=None) -> float:
        dEdt = self.comp_dEdt(self._to._sym_t_norm[intt])
        e    = self.comp_E(self._to._sym_t_norm[intt])
        if intq is not None:
            dvdt = self.comp_dvdt(intq=intq)
            v    = self.V[intq]
            return dEdt * (v - self.V_ref) + e * dvdt
        elif V is not None and q_i is not None and q_o is not None:
            return dEdt(V - self.V_ref) + e * (q_i - q_o)
        else: 
            raise Exception("Input case not covered.")
        
class OdeModel():
    def __init__(self, time_setup_dict) -> None:
        self.time_object = TimeClass(time_setup_dict=time_setup_dict)
        self._state_variable_dict = dict()
    
    def connect_modules(self, 
                        module1:Chamber, 
                        module2:Chamber,
                        pvariable:StateVariable = None,
                        qvariable:StateVariable = None,
                        plabel:str=None, 
                        qlabel:str=None
                        ) ->None:   
        """
        Method for connecting chamber modules.
        
        Inputs
        ------
        module1 (Chamber) : upstream chamber
        module2 (Chamber) : downstream chamber
        plabel (str) : new name for the shared pressure state variable
        qlabel (str) : new name for the shared flow state variable
        """
        module2._Q_i = module1._Q_o
        module2._P_i = module1._P_o
        # module2._Q_i._u = module1.Q_o
        # module2._P_i._u = module1.P_o
        if plabel is not None:
            module1._P_o.set_name(plabel)
            self._state_variable_dict[plabel] = module1._P_o
            print(plabel + ' ' + module1._P_o.name)
        if qlabel is not None:
            module1._Q_o.set_name(qlabel)
            self._state_variable_dict[qlabel] = module1._Q_o
            print(qlabel + ' ' + module1._Q_o.name)
            
    @property
    def state_variable_dict(self):
        return self._state_variable_dict

    
class NaghaviModel(OdeModel):
    def __init__(self, time_setup_dict,) -> None:
        super().__init__(time_setup_dict)
        
        
        # Defining the aorta object
        self.ao = Rc_component( name='Aorta',
                                time_object=self.time_object, 
                                r = 1.0,
                                c = 1.0, 
                                v_ref = 1.0,
                                  )
        
        # Defining the arterial system object
        self.art = Rc_component(name='Arteries',
                                time_object=self.time_object,
                                r = 1.0,
                                c = 1.0,
                                v_ref= 1.0,
                                )
        
        # Defining the venous system object
        self.ven = Rc_component(name='VenaCava',
                                time_object=self.time_object,
                                r = 1.0,
                                c = 1.0,
                                v_ref= 1.0,
                                )
        
        # Defining the aortic valve object
        self.av  = Valve_non_ideal(  name='AorticValve',
                                     time_object=self.time_object,
                                     r=1.0,
                                     max_func=relu_max
                                     )
        
        # Defining the mitral valve object
        self.mv = Valve_non_ideal(  name='MitralValve',
                                    time_object=self.time_object,
                                    r=1.0,
                                    max_func=relu_max
                                    )
        
        # Defining the left atrium activation function
        la_af = lambda t: activation_function_1(t=t, 
                                                t_max=1.0, 
                                                t_tr=1.0, 
                                                tau=1.0
                                                )
        # Defining the left atrium class
        self.la = HC_constant_elastance(name='LeftAtrium',
                                        time_object=self.time_object,
                                        E_pas=1.0,
                                        E_act=1.0,
                                        V_ref=1.0,
                                        activation_function_template=la_af
                                        )
        
        # Defining the left ventricle activation function
        lv_af = lambda t: activation_function_1(t=t,
                                                t_max=1.0,
                                                t_tr=1.0,
                                                tau=1.0
                                                )
        self.lv = HC_constant_elastance(name='LeftVentricle', 
                                        time_object=self.time_object,
                                        E_pas=1.0,
                                        E_act=1.0,
                                        V_ref=1.0,
                                        activation_function_template=lv_af
                                        )
        
        
        # connect the left ventricle class to the aortic valve
        self.connect_modules(self.lv,  self.av,  plabel='p_lv',   qlabel='q_av')
        # connect the aortic valve to the aorta
        self.connect_modules(self.av,  self.ao,  plabel='p_ao',   qlabel='q_av')
        # connect the aorta to the arteries
        self.connect_modules(self.ao,  self.art, plabel= 'p_art', qlabel= 'q_ao')
        # connect the arteries to the veins
        self.connect_modules(self.art, self.ven, plabel= 'p_ven', qlabel= 'q_art')
        # connect the veins to the left atrium
        self.connect_modules(self.ven, self.la,  plabel= 'p_la',  qlabel='q_ven')
        # connect the left atrium to the mitral valve
        self.connect_modules(self.la,  self.mv,  plabel= 'p_la',  qlabel='q_mv')
        # connect the mitral valve to the left ventricle
        self.connect_modules(self.mv,  self.lv,  plabel='p_lv' ,  qlabel='q_mv')
        
        
        
        