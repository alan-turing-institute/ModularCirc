import unittest
import numpy as np
import json
from ModularCirc.Models.OdeModel import OdeModel
from ModularCirc.Solver import Solver
from ModularCirc.Models.KorakianitisMixedModel import KorakianitisMixedModel
from ModularCirc.Models.KorakianitisMixedModel_parameters import KorakianitisMixedModel_parameters

class TestSolver(unittest.TestCase):

    def setUp(self):

        # Set a random seed for reproducibility
        np.random.seed(42)

        # Define the time setup dictionary
        self.time_setup_dict = {
            'name': 'TimeTest',
            'ncycles': 40,
            'tcycle': 1.0,
            'dt': 0.001,
            'export_min': 1
        }

        # Initialize the parameter object
        self.parobj = KorakianitisMixedModel_parameters()

        # Initialize the model
        self.model = KorakianitisMixedModel(time_setup_dict=self.time_setup_dict, parobj=self.parobj, suppress_printing=True)
 
        # Initialize the solver
        self.solver = Solver(model=self.model)

        # Setup the solver
        self.solver.setup(suppress_output=True, method='LSODA')        

    def test_solver_initialization(self):
        # Verify solver is an instance of Solver
        self.assertIsInstance(self.solver, Solver)
        # Verify model is correctly assigned
        self.assertEqual(self.solver.model, self.model)

    def test_solver_setup(self):

        # Verify the setup attributes
        self.assertEqual(self.solver._method, 'LSODA')
        self.assertTrue(self.solver._optimize_secondary_sv is False)

        # Verify the generated functions
        self.assertIsNotNone(self.solver.pv_dfdt_global)
        self.assertIsNotNone(self.solver.s_u_update)
        self.assertIsNotNone(self.solver.optimize)     
        self.assertIsNotNone(self.solver.initialize_by_function)                

        # Test initialize_by_function():

        # Verify that initialize_by_function accepts the expected input, and returns the expected output
        # Load the expected values from an npy file
        expected_input = np.load('tests/inputs_for_tests/asd_first_row.npy')

        # Verify the function returns the output in the right data type
        self.assertIsInstance(self.solver.initialize_by_function(y=expected_input), np.ndarray)

        # Generate a random input that is not the expected input
        random_input = np.random.rand(1, 1)
        # Verify that the function does not accept the random input
        with self.assertRaises(IndexError):
            self.solver.initialize_by_function(y=random_input)

        # Test optimize():

        # Verify that optimize() accepts the expected input
        # Load the expected values from an npz file
        expected_input = np.load('tests/inputs_for_tests/inputs_for_optimize.npz')

        # Verify the function can run with the expected input
        self.solver.optimize(y=expected_input['y_temp'], keys=expected_input['keys4'])

        # Test pv_dfdt_update():

        # Verify that pv_dfdt_update() accepts the expected input
        # Load the expected values from an npy file
        y0 = np.load('tests/inputs_for_tests/inputs_for_pv_dfdt_update.npy')

        # Verify the function can run with the expected input
        pv_dfdt_result = self.solver.pv_dfdt_global(t=0, y=y0) 

        # Verify the output matches the expected output
        expected_output = np.load('tests/expected_outputs/pv_dfdt_update_expected_output.npy')
        np.testing.assert_allclose(pv_dfdt_result, expected_output)

        # Test s_u_update():

        # Verify that s_u_update() accepts the expected input
        # Load the expected values from an npy file
        y_temp = np.load('tests/inputs_for_tests/inputs_for_s_u_update.npy')

        # Verify the function can run with the expected input
        s_u_result = self.solver.s_u_update(t=0, y=y_temp)

        # Verify the output matches the expected output
        expected_output = np.load('tests/expected_outputs/s_u_update_expected_output.npy')
        np.testing.assert_allclose(s_u_result, expected_output)


    def test_solver_solve(self):

        # Solve the system
        self.solver.solve()

        # Verify the solver converged
        self.assertTrue(self.solver.converged or self.solver._Nconv is not None)

        # Load expected values from a JSON file
        with open('tests/expected_outputs/KorakianitisMixedModel_expected_output.json', 'r') as f:
            self.expected_values = json.load(f)

        # Redefine tind based on how many heart cycle have actually been necessary to reach steady state
        self.tind_fin  = np.arange(start=self.model.time_object.n_t-self.model.time_object.n_c * self.model.time_object.export_min,
                                   stop=self.model.time_object.n_t)
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