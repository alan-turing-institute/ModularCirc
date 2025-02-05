import unittest
import numpy as np
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
        
        for key, value in self.model.commponents.items():
            self.initial_values[key] = {
                'V': value.V.values[self.tind_init].mean(),
                'P_i': value.P_i.values[self.tind_init].mean(),
                'Q_i': value.Q_i.values[self.tind_init].mean()
            }

        # Expected values from a previous run
        self.expected_values = {'la': {'V': 126.38884973, 'P_i': 1.9049962, 'Q_i': 85.55031592}, 'mi': {'V': 0.0, 'P_i': 1.9049962851, 'Q_i': 85.1636944244}, 'lv': {'V': 78.92264366, 'P_i': 35.6800612, 'Q_i': 85.16369442}, 'ao': {'V': 0.0, 'P_i': 35.6800612, 'Q_i': 85.72479065}, 'sas': {'V': 7.9275133, 'P_i': 99.0939162631, 'Q_i': 85.724790652}, 'sat': {'V': 158.13864426, 'P_i': 98.83665266, 'Q_i': 85.74867979}, 'svn': {'V': 145.49804915, 'P_i': 7.09746581, 'Q_i': 85.73761272}, 'ra': {'V': 53.83474153, 'P_i': 0.73421251, 'Q_i': 84.8433774}, 'ti': {'V': 0.0, 'P_i': 0.73421251, 'Q_i': 84.58031928}, 'rv': {'V': 66.10678226, 'P_i': 11.48872347, 'Q_i': 84.58031928}, 'po': {'V': 0.0, 'P_i': 11.48872347, 'Q_i': 83.979334974}, 'pas': {'V': 5.16689938, 'P_i': 28.704996546, 'Q_i': 83.97933499}, 'pat': {'V': 108.44076581847, 'P_i': 28.53704363, 'Q_i': 83.97564173}, 'pvn': {'V': 49.57511092, 'P_i': 2.41829809, 'Q_i': 84.254321165}}


    def test_model_initialization(self):
        self.assertIsInstance(self.model, KorakianitisMixedModel)
        self.assertTrue(hasattr(self.solver.model, 'commponents'))
        self.assertIn('lv', self.solver.model.commponents)
        self.assertEqual(self.solver.model.commponents['lv'].E_pas, self.parobj.components['lv']['E_pas'])
        self.assertEqual(self.solver.model.commponents['ao'].CQ, self.parobj.components['ao']['CQ'])

    def test_solver_initialization(self):
        self.assertIsInstance(self.solver, Solver)
        self.assertEqual(self.solver.model, self.model)

    def test_solver_run(self):
        self.solver.solve()
        self.assertTrue(len(self.solver.model.commponents['lv'].V.values) > 0)
        self.assertTrue(len(self.solver.model.commponents['lv'].P_i.values) > 0)
        
        self.tind_fin  = np.arange(start=self.model.time_object.n_t-self.model.time_object.n_c * self.model.time_object.export_min,
                               stop =self.model.time_object.n_t)

        new_dict = {}
        for key, value in self.model.commponents.items():

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
