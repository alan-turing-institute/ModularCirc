from .OdeModel import OdeModel
from .MixedHeartMaynard4eWindkessel_parameters import MixedHeartMaynard4eWindkessel_parameters as MHM4W_parobj
from .ParametersObject import ParametersObject as po
from ..Components import Rlc_component, Valve_maynard,  \
    HC_mixed_elastance, R_component, Valve_non_ideal, \
        HC_constant_elastance, Valve_simple_bernoulli

FULL_NAMES =[
    'LeftA',
    'MiValve',
    'LeftV',
    'AoV',
    'SysArtImp',
    'SysArt',
    'SysCap',
    'SysVen',
    'RightA',
    'TriValve',
    'RightV',
    'PulV',
    'PulArtImp',
    'PulArt',
    'PulCap',
    'PulVen',
]

class MixedHeartMaynard4eWindkessel(OdeModel):
    def __init__(self, time_setup_dict, parobj:po=MHM4W_parobj) -> None:
        super().__init__(time_setup_dict)
        self.name = 'MixedHeartMaynard4eWindkessel'
        
        print(parobj)
        
        # The components...
        for key, name in zip(parobj.components.keys(), FULL_NAMES):
            if key in parobj._vessels: 
                class_ = Rlc_component
            elif key in parobj._imp or key in parobj._cap:
                class_ =  R_component
            elif key in parobj._valves:
                class_ = Valve_simple_bernoulli # Valve_non_ideal # Valve_maynard # Valve_simple_bernoulli
            elif key in parobj._chambers:
                class_ = HC_constant_elastance # HC_mixed_elastance HC_constant_elastance
            else:
                raise Exception(f'Component name {key} not in the model list.')
            self.components[key] = class_(name=name,
                                    time_object=self.time_object, 
                                    **parobj[key].to_dict())
            
            if key not in parobj._valves + parobj._cap + parobj._imp: 
                self.set_v_sv(key)
            # else:
            #     self.set_phi_sv(key)
            self.components[key].setup()
            
        self.connect_modules(self.components['lv'],
                            self.components['ao'],
                            plabel='p_lv',
                            qlabel='q_ao',
                            )
        self.connect_modules(self.components['ao'],
                            self.components['sai'],
                            plabel='p_sa',
                            qlabel='q_ao')
        self.connect_modules(self.components['sai'],
                            self.components['sa'],
                            plabel='pi_sa',
                            qlabel='q_ao',
                            qvariable=self.components['ao']._Q_o)
        self.connect_modules(self.components['sa'],
                            self.components['sc'],
                            plabel='p_sc',
                            qlabel='q_sa')
        self.connect_modules(self.components['sc'],
                            self.components['sv'],
                            plabel='p_sv',
                            qlabel='q_sa',
                            qvariable=self.components['sa']._Q_o)
        self.connect_modules(self.components['sv'],
                            self.components['ra'],
                            plabel='p_ra',
                            qlabel='q_sv')
        self.connect_modules(self.components['ra'],
                            self.components['ti'],
                            plabel='p_ra',
                            qlabel='q_ti')
        self.connect_modules(self.components['ti'],
                            self.components['rv'],
                            plabel='p_rv',
                            qlabel='q_ti')
        self.connect_modules(self.components['rv'],
                            self.components['po'],
                            plabel='p_rv',
                            qlabel='q_po')
        self.connect_modules(self.components['po'],
                            self.components['pai'],
                            plabel='p_pa',
                            qlabel='q_po')
        self.connect_modules(self.components['pai'],
                            self.components['pa'],
                            plabel='pi_pa',
                            qlabel='q_po',
                            qvariable=self.components['po']._Q_o)
        self.connect_modules(self.components['pa'],
                            self.components['pc'],
                            plabel='p_pc',
                            qlabel='q_pa')
        self.connect_modules(self.components['pc'],
                            self.components['pv'],
                            plabel='p_pv',
                            qlabel='q_pa',
                            qvariable=self.components['pa']._Q_o)
        self.connect_modules(self.components['pv'],
                            self.components['la'],
                            plabel='p_la',
                            qlabel='q_pv')
        self.connect_modules(self.components['la'],
                            self.components['mi'],
                            plabel='p_la',
                            qlabel='q_mi')
        self.connect_modules(self.components['mi'],
                            self.components['lv'],
                            plabel='p_lv',
                            qlabel='q_mi')
        for component in self.components.values():
            component.setup()
            
    def set_phi_sv(self, comp_key:str) -> None:
        phi_key = 'phi_' + comp_key
        self._state_variable_dict[phi_key] = self.components[comp_key]._PHI
        self._state_variable_dict[phi_key].set_name(phi_key)
        self.all_sv_data[phi_key] = self.components[comp_key].PHI