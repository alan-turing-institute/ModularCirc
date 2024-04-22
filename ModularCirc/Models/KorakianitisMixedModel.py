from .OdeModel import OdeModel
from .KorakianitisModel_parameters import KorakianitisModel_parameters as k2006
from .ParametersObject import ParametersObject as po
from ..Components import Rlc_component, Valve_simple_bernoulli, HC_mixed_elastance

FULL_NAMES =[
    'LeftA',
    'MiValve',
    'LeftV',
    'AoV',
    'SysAoSin',
    'SysArt',
    'SysVen',
    'RightA',
    'TriValve',
    'RightV',
    'PulV',
    'PulArtSin',
    'PulArt',
    'PulVen',
]

class KorakianitisMixedModel(OdeModel):
    def __init__(self, time_setup_dict, parobj:po=k2006, supress_printing:bool=False) -> None:
        super().__init__(time_setup_dict)
        self.name = 'KorakianitisModel'
        
        if not supress_printing: print(parobj)
        
        # The components...
        for key, name in zip(parobj.components.keys(), FULL_NAMES):
            if key in parobj._vessels: 
                class_ = Rlc_component
            elif key in parobj._valves:
                class_ = Valve_simple_bernoulli
            elif key in parobj._chambers:
                class_ = HC_mixed_elastance
            else:
                raise Exception(f'Component name {key} not in the model list.')
            self.commponents[key] = class_(name=name,
                                    time_object=self.time_object, 
                                    **parobj[key].to_dict())
            if key not in parobj._valves: 
                self.set_v_sv(key)
            # else:
            #     self.set_phi_sv(key)
            self.commponents[key].setup()
            
        self.connect_modules(self.commponents['lv'],
                             self.commponents['ao'],
                             plabel='p_lv',
                             qlabel='q_ao')
        self.connect_modules(self.commponents['ao'],
                             self.commponents['sas'],
                             plabel='p_sas',
                             qlabel='q_ao')
        self.connect_modules(self.commponents['sas'],
                             self.commponents['sat'],
                             plabel='p_sat',
                             qlabel='q_sas')
        self.connect_modules(self.commponents['sat'],
                             self.commponents['svn'],
                             plabel='p_svn',
                             qlabel='q_sat')
        self.connect_modules(self.commponents['svn'],
                             self.commponents['ra'],
                             plabel='p_ra',
                             qlabel='q_svn')
        self.connect_modules(self.commponents['ra'],
                             self.commponents['ti'],
                             plabel='p_ra',
                             qlabel='q_ti')
        self.connect_modules(self.commponents['ti'],
                             self.commponents['rv'],
                             plabel='p_rv',
                             qlabel='q_ti')
        self.connect_modules(self.commponents['rv'],
                             self.commponents['po'],
                             plabel='p_rv',
                             qlabel='q_po')
        self.connect_modules(self.commponents['po'],
                             self.commponents['pas'],
                             plabel='p_pas',
                             qlabel='q_po')
        self.connect_modules(self.commponents['pas'],
                             self.commponents['pat'],
                             plabel='p_pat',
                             qlabel='q_pas')
        self.connect_modules(self.commponents['pat'],
                             self.commponents['pvn'],
                             plabel='p_pvn',
                             qlabel='q_pat')
        self.connect_modules(self.commponents['pvn'],
                             self.commponents['la'],
                             plabel='p_la',
                             qlabel='q_pvn')
        self.connect_modules(self.commponents['la'],
                             self.commponents['mi'],
                             plabel='p_la',
                             qlabel='q_mi')
        self.connect_modules(self.commponents['mi'],
                             self.commponents['lv'],
                             plabel='p_lv',
                             qlabel='q_mi')
        
        for component in self.commponents.values():
            component.setup()
            
    # def set_phi_sv(self, comp_key:str) -> None:
    #     phi_key = 'phi_' + comp_key
    #     self._state_variable_dict[phi_key] = self.commponents[comp_key]._PHI
    #     self._state_variable_dict[phi_key].set_name(phi_key)
    #     self.all_sv_data[phi_key] = self.commponents[comp_key].PHI