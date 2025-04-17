from .OdeModel import OdeModel
from .KorakianitisMixedModel_parameters import KorakianitisMixedModel_parameters, TIME_SETUP_DICT
from .KorakianitisMixedModel_parameters import TIME_SETUP_DICT as TEMPLATE_TIME_SETUP_DICT
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
    def __init__(self, time_setup_dict, parobj:po=KorakianitisMixedModel_parameters, suppress_printing:bool=False) -> None:
        super().__init__(time_setup_dict)
        self.name = 'KorakianitisModel'

        if not suppress_printing: print(parobj)

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
            self.components[key] = class_(name=name,
                                    time_object=self.time_object,
                                    **parobj[key].to_dict())
            if key not in parobj._valves:
                self.set_v_sv(key)
            # else:
            #     self.set_phi_sv(key)
            self.components[key].setup()

        self.connect_modules(self.components['lv'],
                             self.components['ao'],
                             plabel='p_lv',
                             qlabel='q_ao')
        self.connect_modules(self.components['ao'],
                             self.components['sas'],
                             plabel='p_sas',
                             qlabel='q_ao')
        self.connect_modules(self.components['sas'],
                             self.components['sat'],
                             plabel='p_sat',
                             qlabel='q_sas')
        self.connect_modules(self.components['sat'],
                             self.components['svn'],
                             plabel='p_svn',
                             qlabel='q_sat')
        self.connect_modules(self.components['svn'],
                             self.components['ra'],
                             plabel='p_ra',
                             qlabel='q_svn')
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
                             self.components['pas'],
                             plabel='p_pas',
                             qlabel='q_po')
        self.connect_modules(self.components['pas'],
                             self.components['pat'],
                             plabel='p_pat',
                             qlabel='q_pas')
        self.connect_modules(self.components['pat'],
                             self.components['pvn'],
                             plabel='p_pvn',
                             qlabel='q_pat')
        self.connect_modules(self.components['pvn'],
                             self.components['la'],
                             plabel='p_la',
                             qlabel='q_pvn')
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

    # def set_phi_sv(self, comp_key:str) -> None:
    #     phi_key = 'phi_' + comp_key
    #     self._state_variable_dict[phi_key] = self.components[comp_key]._PHI
    #     self._state_variable_dict[phi_key].set_name(phi_key)
    #     self.all_sv_data[phi_key] = self.components[comp_key].PHI
