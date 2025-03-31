import unittest
import numpy as np
import json
from ModularCirc.Models.OdeModel import OdeModel
from ModularCirc.Solver import Solver
from ModularCirc.Models.KorakianitisMixedModel import KorakianitisMixedModel
from ModularCirc.Models.KorakianitisMixedModel_parameters import KorakianitisMixedModel_parameters

class TestSolver(unittest.TestCase):
    """
    TestSolver is a unittest.TestCase class designed to test the functionality of the Solver class and its 
    integration with the KorakianitisMixedModel.

    Methods
    -------
    setUp():
        Initializes the test environment, including setting a random seed, defining the time setup dictionary, 
        initializing the parameter object, model, and solver, and setting up the solver.
    test_solver_initialization():
        Verifies that the solver is correctly initialized and that the model is correctly assigned to the solver.
    test_solver_setup():
        Verifies the setup attributes of the solver and ensures that the generated functions are not None.
    test_initialize_by_function():
        Tests the initialize_by_function() method of the solver, ensuring it accepts the expected input and 
        returns the expected output.
    test_optimize():
        Tests the optimize() method of the solver, ensuring it accepts the expected input and can run with it.
    test_pv_dfdt_update():
        Tests the pv_dfdt_update() method of the solver, ensuring it accepts the expected input and the output 
        matches the expected output.
    test_s_u_update():
        Tests the s_u_update() method of the solver, ensuring it accepts the expected input and the output 
        matches the expected output.
    test_solver_solve():
        Tests the solve() method of the solver, ensuring the solver converges and the output matches the 
        expected values.
    """

    def setUp(self):
        """
        Set up the test environment for the Solver tests.
        This method initializes the necessary components for testing the Solver class:
        - Sets a random seed for reproducibility.
        - Defines the time setup dictionary with parameters for the simulation.
        - Initializes the parameter object using KorakianitisMixedModel_parameters.
        - Initializes the KorakianitisMixedModel with the time setup dictionary and parameter object.
        - Initializes the Solver with the model.
        - Sets up the solver with the specified method and suppresses output.
        Attributes:
            time_setup_dict (dict): Dictionary containing time setup parameters.
            parobj (KorakianitisMixedModel_parameters): Parameter object for the model.
            model (KorakianitisMixedModel): The initialized model for the simulation.
            solver (Solver): The initialized solver for the model.
        """

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
        """
        Test the initialization of the solver.

        This test verifies that the solver is correctly initialized as an instance of the 
        Solver class and that the model is correctly assigned to the solver.

        Assertions:
            - The solver is an instance of the Solver class.
            - The model assigned to the solver is the expected model.
        """
        # Verify solver is an instance of Solver
        self.assertIsInstance(self.solver, Solver)
        # Verify model is correctly assigned
        self.assertEqual(self.solver.model, self.model)


    def test_solver_setup(self):
        """
        Test the setup of the solver.

        This test verifies that the solver's setup attributes and generated functions
        are correctly initialized. It checks the following:
        - The solver's method is set to 'LSODA'.
        - The solver's optimize_secondary_sv attribute is set to False.
        - The solver's pv_dfdt_global function is not None.
        - The solver's s_u_update function is not None.
        - The solver's optimize function is not None.
        - The solver's initialize_by_function function is not None.
        """

        # Verify the setup attributes
        self.assertEqual(self.solver._method, 'LSODA')
        self.assertTrue(self.solver._optimize_secondary_sv is False)

        # Verify the generated functions
        self.assertIsNotNone(self.solver.pv_dfdt_global)
        self.assertIsNotNone(self.solver.s_u_update)
        self.assertIsNotNone(self.solver.optimize)     
        self.assertIsNotNone(self.solver.initialize_by_function)                


    def test_initialize_by_function(self):
        """
        Test the `initialize_by_function` method of the solver.

        This test verifies the following:
        - The `initialize_by_function` method accepts the expected input and returns the expected output.
        - The output of the `initialize_by_function` method is of type `np.ndarray`.
        - The `initialize_by_function` method raises an `IndexError` when provided with a random input that is not the expected input.

        Test steps:
        1. Load the expected input values from an npy file.
        2. Verify that the `initialize_by_function` method returns an output of type `np.ndarray` when provided with the expected input.
        3. Generate a random input that is not the expected input.
        4. Verify that the `initialize_by_function` method raises an `IndexError` when provided with the random input.
        """

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


    def test_optimize(self):
        """
        Test the optimize() method of the Solver class.

        This test verifies that the optimize() method accepts the expected input
        and can run with the provided input data.

        Steps:
        1. Load the expected input values from a .npz file.
        2. Verify that the optimize() method can run with the expected input.

        Expected input:
        - y_temp: Temporary y values for optimization. Called y_temp to reflect the 
                    name of the variable in the optimize() method.
        - keys4: Keys required for optimization. Called keys4 to reflect the name of the
                    variable in the optimize() method.  

        Raises:
            AssertionError: If the optimize() method does not accept the expected input
                            or fails to run with the provided input data.
        """

        # Test optimize():

        # Verify that optimize() accepts the expected input
        # Load the expected values from an npz file
        expected_input = np.load('tests/inputs_for_tests/inputs_for_optimize.npz')

        # Verify the function can run with the expected input
        self.solver.optimize(y=expected_input['y_temp'], keys=expected_input['keys4'])


    def test_pv_dfdt_update(self):
        """
        Test the `pv_dfdt_update` method of the solver.

        This test verifies that the `pv_dfdt_update` method:
        1. Accepts the expected input.
        2. Can run with the expected input without errors.
        3. Produces output that matches the expected output.

        Steps:
        - Load the initial conditions from a .npy file.
        - Run the `pv_dfdt_update` method with the loaded initial conditions.
        - Load the expected output from a .npy file.
        - Compare the method's output to the expected output using `np.testing.assert_allclose`.

        Raises:
            AssertionError: If the output of `pv_dfdt_update` does not match the expected output.
        """

        # Test pv_dfdt_update():

        # Verify that pv_dfdt_update() accepts the expected input
        # Load the expected values from an npy file
        y0 = np.load('tests/inputs_for_tests/inputs_for_pv_dfdt_update.npy')

        # Verify the function can run with the expected input
        pv_dfdt_result = self.solver.pv_dfdt_global(t=0, y=y0) 

        # Verify the output matches the expected output
        expected_output = np.load('tests/expected_outputs/pv_dfdt_update_expected_output.npy')
        np.testing.assert_allclose(pv_dfdt_result, expected_output)


    def test_s_u_update(self):
        """
        Test the s_u_update() method of the solver.

        This test verifies the following:
        1. The s_u_update() method accepts the expected input.
        2. The s_u_update() method can run with the expected input.
        3. The output of the s_u_update() method matches the expected output.

        Steps:
        - Load the expected input values from a .npy file.
        - Call the s_u_update() method with the loaded input.
        - Load the expected output values from a .npy file.
        - Compare the actual output with the expected output using np.testing.assert_allclose.

        Raises:
            AssertionError: If the actual output does not match the expected output.
        """

        # Test s_u_update():

        # Verify that s_u_update() accepts the expected input
        # Load the expected values from an npy file
        y_temp = np.load('tests/inputs_for_tests/inputs_for_s_u_update.npy')

        # Verify the function can run with the expected input
        s_u_result = self.solver.s_u_update(t=0.0, y=y_temp)

        # Verify the output matches the expected output
        expected_output = np.load('tests/expected_outputs/s_u_update_expected_output.npy')
        np.testing.assert_allclose(s_u_result, expected_output)


    def test_solver_solve(self):
        """
        Test the `solve` method of the solver.

        This test performs the following steps:
        1. Solves the system using the solver.
        2. Verifies that the solver has converged.
        3. Loads expected values from a JSON file.
        4. Redefines the time indices based on the number of heart cycles necessary to reach steady state.
        5. Retrieves the component state variables, computes the mean values during the last cycle, and stores them in a new solution dictionary.
        6. Compares the new solution dictionary values with the expected values and asserts that they are within an acceptable tolerance.

        Raises:
            AssertionError: If the solver did not converge or if the computed values do not match the expected values within the tolerance.
        """

        # Solve the system
        self.solver.solve()

        # Verify the solver converged
        self.assertTrue(self.solver.converged or self.solver._Nconv is not None)

        # Load expected values from a JSON file
        with open('tests/expected_outputs/KorakianitisMixedModel_expected_output.json', 'r') as f:
            self.expected_values = json.load(f)

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