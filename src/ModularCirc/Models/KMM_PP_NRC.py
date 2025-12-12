from .OdeModel import OdeModel
from .KMM_PP_NRC_parameters import KMM_PP_NRC_parameters
from .ParametersObject import ParametersObject as po
from ..Components import Rlc_component, Valve_simple_bernoulli, HC_mixed_elastance, Rc_nonlinear_component


FULL_NAMES = [
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
],   # pulmonary valve

class KMM_PP_NRC(OdeModel):
    def __init__(self, time_setup_dict, parobj=KMM_PP_NRC_parameters, suppress_printing:bool=False) -> None:
        super().__init__(time_setup_dict)
        self.name = 'KMM_PP_NRC'

        if not suppress_printing: print(parobj)

        # The components...
        for key, name in zip(parobj.components.keys(), FULL_NAMES):
            if key in parobj._vessels:
                class_ = Rlc_component
            elif key in parobj._nonlinear_vessels:
                class_ = Rc_nonlinear_component
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