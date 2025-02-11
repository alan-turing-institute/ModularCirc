import unittest
import numpy as np
import json
from ModularCirc.Models.KorakianitisMixedModel import KorakianitisMixedModel
from ModularCirc.Models.KorakianitisMixedModel_parameters import KorakianitisMixedModel_parameters
from ModularCirc.Solver import Solver


class TestKorakianitisMixedModel(unittest.TestCase):

    def setUp(self):
        self.time_setup_dict = {
            'name': 'TimeTest',
            'ncycles': 40,
            'tcycle': 1.0,
            'dt': 0.001,
            'export_min': 1
        }
        self.parobj = KorakianitisMixedModel_parameters()
        self.model = KorakianitisMixedModel(time_setup_dict=self.time_setup_dict, parobj=self.parobj, suppress_printing=True)
        self.solver = Solver(model=self.model)
        self.solver.setup(suppress_output=True, method='LSODA')

        self.initial_values = {}
        
        self.tind_init  = np.arange(start=self.model.time_object.n_t-self.model.time_object.n_c * self.model.time_object.export_min,
                               stop =self.model.time_object.n_t)
        
        for key, value in self.model.components.items():
            self.initial_values[key] = {
                'V': value.V.values[self.tind_init].mean(),
                'P_i': value.P_i.values[self.tind_init].mean(),
                'Q_i': value.Q_i.values[self.tind_init].mean()
            }

        # Load expected values from a JSON file
        with open('tests/expected_outputs/KorakianitisMixedModel_expected_output.json', 'r') as f:
            self.expected_values = json.load(f)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, KorakianitisMixedModel)
        self.assertTrue(hasattr(self.solver.model, 'components'))
        self.assertIn('lv', self.solver.model.components)
        self.assertEqual(self.solver.model.components['lv'].E_pas, self.parobj.components['lv']['E_pas'])
        self.assertEqual(self.solver.model.components['ao'].CQ, self.parobj.components['ao']['CQ'])

    def test_solver_initialization(self):
        self.assertIsInstance(self.solver, Solver)
        self.assertEqual(self.solver.model, self.model)

    def test_solver_run(self):
        self.solver.solve()
        self.assertTrue(len(self.solver.model.components['lv'].V.values) > 0)
        self.assertTrue(len(self.solver.model.components['lv'].P_i.values) > 0)
        
        self.tind_fin  = np.arange(start=self.model.time_object.n_t-self.model.time_object.n_c * self.model.time_object.export_min,
                                   stop=self.model.time_object.n_t)

        new_dict = {}
        for key, value in self.model.components.items():

            new_dict[key] = {
                'V': value.V.values[self.tind_fin].mean(),
                'P_i': value.P_i.values[self.tind_fin].mean(),
                'Q_i': value.Q_i.values[self.tind_fin].mean()
            }

        # Check that the values have changed wrt the initial values
        self.assertFalse(self.initial_values == new_dict)

        # Check that the values are the same as the expected values
        expected_ndarray = np.array([self.expected_values[key1][key2]  for key1 in new_dict.keys() for key2 in new_dict[key1].keys()])
        new_ndarray      = np.array([new_dict[key1][key2]              for key1 in new_dict.keys() for key2 in new_dict[key1].keys()])
        test_ndarray     = np.where(np.abs(expected_ndarray) > 1e-6, np.abs((expected_ndarray - new_ndarray) / expected_ndarray),  np.abs((expected_ndarray - new_ndarray)))
        self.assertTrue((test_ndarray < 1e-3).all())


if __name__ == '__main__':
    unittest.main()
