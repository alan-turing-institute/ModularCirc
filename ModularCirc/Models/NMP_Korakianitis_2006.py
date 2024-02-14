from .NaghaviModelParameters import NaghaviModelParameters as nmp
from ..HelperRoutines import activation_function_2

class NMP_Korakianitis_2006(nmp):
    """
    Intro
    -----
    Alteration fo the Naghavi Model Parameters based on Korakianitis and Shi (2006)

    Derived  from:
    ---
        nmp (class): NagaviModelParameters class
    """
    def __init__(self) -> None:
        super().__init__()
        
        mmhg = 133.32
        
        self.set_activation_function('lv', activation_func=activation_function_2)
        self.set_chamber_comp('lv',
                              E_pas=mmhg * 0.1,
                              E_act=mmhg * 2.5,
                              V_ref=5.0,
                              t_max=30,
                              tau=45)
        
        self.set_activation_function('la', activation_func=activation_function_2)
        self.set_chamber_comp('la',
                              E_pas=mmhg * 0.15,
                              E_act=mmhg * 0.25,
                              V_ref=4.0,
                              t_max=9.0,
                              tau=18.0,
                              delay=17.0)
        
        self.set_rlc_comp('ao', 
                          r=0.003*mmhg,
                          c=0.08/mmhg,
                          l=0.000062*mmhg)
        self.set_rlc_comp('art',
                          r=(0.05 + 0.5 + 0.52) * mmhg,
                          c=1.6/mmhg,
                          l=0.0017*mmhg)
        self.set_rc_comp('ven',
                         r=0.075*mmhg,
                         c=20.5/mmhg)
        
        self.set_valve_comp('av',r=0.0003*mmhg)
        self.set_valve_comp('mv',r=0.00003*mmhg)
        
