import unittest
import numpy as np
import json
from ModularCirc.Models.KorakianitisMixedModel import KorakianitisMixedModel
from ModularCirc.Models.KorakianitisMixedModel_parameters import KorakianitisMixedModel_parameters
from ModularCirc.Solver import Solver


class TestKorakianitisMixedModel(unittest.TestCase):

    def setUp(self):
        # Define the duration of the simulation (no of cycles), duration of the cycle, maximum time step size, and minimum number of cycles to run 
        self.time_setup_dict = {
            'name': 'TimeTest',
            'ncycles': 40,
            'tcycle': 1.0,
            'dt': 0.001,
            'export_min': 1
        }
        # Initializing the parameter object
        self.parobj = KorakianitisMixedModel_parameters()
        # Initializing the model 
        self.model = KorakianitisMixedModel(time_setup_dict=self.time_setup_dict, parobj=self.parobj, suppress_printing=True)
        # Initializing the solver
        self.solver = Solver(model=self.model)
        # Solver is being setup: switching off console printing and setting the solver method to "LSODA"
        self.solver.setup(suppress_output=True, method='LSODA')

        # Load expected values from a JSON file
        with open('tests/expected_outputs/KorakianitisMixedModel_expected_output.json', 'r') as f:
            self.expected_values = json.load(f)

    def test_model_initialization(self):
        '''
        Testing the initialization of the solver, model and parameter objects.
        '''
        # Verify model is an instance of <KorakianitisMixedModel>
        self.assertIsInstance(self.model, KorakianitisMixedModel)
        # Verify model has attribute <components>
        self.assertTrue(hasattr(self.solver.model, 'components'))
        # Verify <lv> is a component 
        self.assertIn('lv', self.solver.model.components)
        # Verify correct assignment of parameters from parobj to model
        self.assertEqual(self.solver.model.components['lv'].E_pas, self.parobj.components['lv']['E_pas'])
        self.assertEqual(self.solver.model.components['ao'].CQ, self.parobj.components['ao']['CQ'])

    def test_solver_initialization(self):
        # Verify <solver> is an instance of <Solver>
        self.assertIsInstance(self.solver, Solver)
        # Verify the instance of <model> that is an atribute of <solver> is the same as the original <model>
        self.assertEqual(self.solver.model, self.model)

    def test_solver_run(self):
        '''
        Testing running the solver: all the components (their associated equations) are assembled within a system. The system
        of equations is then passed to solve_ivp (retrieved from scipy).
        '''
        
        # Running the model
        self.solver.solve()

        # Verifying the model changed the state variables stored within components.
        self.assertTrue(len(self.solver.model.components['lv'].V.values) > 0)
        self.assertTrue(len(self.solver.model.components['lv'].P_i.values) > 0)

        # Redefine tind based on how many heart cycle have actually been necessary to reach steady state
        self.tind_fin  = np.arange(start=self.model.time_object.n_t-self.model.time_object.n_c,
                                   stop=(self.model.time_object.n_t))

        # Retrieve the component state variables, compute the mean of the values during the last cycle and store them within
        # the new solution dictionary
        new_dict = {}
        for key, value in self.model.components.items():

            new_dict[key] = {
                'V': value.V.values[self.tind_fin].mean(),
                'P_i': value.P_i.values[self.tind_fin].mean(),
                'Q_i': value.Q_i.values[self.tind_fin].mean()
            }

        # Check that the values are the same as the expected values
        expected_ndarray = np.array([self.expected_values[key1][key2]  for key1 in new_dict.keys() for key2 in new_dict[key1].keys()])
        new_ndarray      = np.array([new_dict[key1][key2]              for key1 in new_dict.keys() for key2 in new_dict[key1].keys()])
        test_ndarray     = np.where(np.abs(expected_ndarray) > 1e-6, np.abs((expected_ndarray - new_ndarray) / expected_ndarray),  np.abs((expected_ndarray - new_ndarray)))
        self.assertTrue((test_ndarray < 1e-3).all())


if __name__ == '__main__':
    unittest.main()
