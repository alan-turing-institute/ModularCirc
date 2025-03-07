import unittest
import numpy as np
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

    def test_solver_initialization(self):
        # Verify solver is an instance of Solver
        self.assertIsInstance(self.solver, Solver)
        # Verify model is correctly assigned
        self.assertEqual(self.solver.model, self.model)

    def test_solver_setup(self):
        
        # Setup the solver
        self.solver.setup(suppress_output=True, method='LSODA')

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
        with self.assertRaises(ValueError):
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
        pv_dfdt_result = self.solver.pv_dfdt_global(y0, t=0) 

        # Verify the output matches the expected output
        expected_output = np.load('tests/expected_outputs/expected_output_pv_dfdt_global.npy')
        np.testing.assert_allclose(pv_dfdt_result, expected_output)


    def test_advance_cycle(self):
        # Setup the solver
        self.solver.setup(suppress_output=True, method='LSODA')
        # Initialize the solution fields
        self.solver._asd.loc[0, self.solver._initialize_by_function.index] = \
            self.solver.initialize_by_function(y=self.solver._asd.loc[0].to_numpy()).T
        # Advance one cycle
        y0 = self.solver._asd.iloc[0, list(self.solver._global_psv_update_fun.keys())].to_list()
        result = self.solver.advance_cycle(y0=y0, cycleID=0)
        # Verify the result
        self.assertFalse(result)

    def test_solver_solve(self):
        # Setup the solver
        self.solver.setup(suppress_output=True, method='LSODA')
        # Solve the system
        self.solver.solve()
        # Verify the solver converged
        self.assertTrue(self.solver.converged or self.solver._Nconv is not None)

if __name__ == '__main__':
    unittest.main()