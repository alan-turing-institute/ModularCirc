import unittest
import numpy as np
import json
import os
import logging
from ModularCirc.Models.KorakianitisMixedModel import KorakianitisMixedModel
from ModularCirc.Models.KorakianitisMixedModel_parameters import KorakianitisMixedModel_parameters
from ModularCirc.Solver import Solver

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define global constants for tolerances
RELATIVE_TOLERANCE = 1e-3

class TestKorakianitisMixedModel(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment for the KorakianitisMixedModel.

        This method initializes the necessary components and configurations
        required for testing the KorakianitisMixedModel. It performs the following:

        - Sets a random seed for reproducibility.
        - Defines the base directory for file paths.
        - Configures the simulation time setup parameters, including the number
          of cycles, cycle duration, time step size, and minimum export cycles.
        - Initializes the parameter object for the KorakianitisMixedModel.
        - Creates an instance of the KorakianitisMixedModel with the specified
          time setup and parameter object, suppressing console output.
        - Initializes the solver for the model and configures it with the
          "LSODA" method, suppressing console output.
        - Loads expected output values from a JSON file for validation during tests.
        - Verifies the existence of the expected output file and raises an
          assertion error if the file is not found.

        Raises:
            AssertionError: If the expected output file is not found.
        """

        # Set a random seed for reproducibility
        np.random.seed(42)

        # Define the base directory for file paths
        self.base_dir = os.path.dirname(__file__)

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
        self.model = KorakianitisMixedModel(time_setup_dict=self.time_setup_dict,
                                            parobj=self.parobj,
                                            suppress_printing=True)

        # Initializing the solver
        self.solver = Solver(model=self.model)

        # Solver is being setup: switching off console printing and setting the solver method to "LSODA"
        self.solver.setup(suppress_output=True, method='LSODA',step=1)

        # Load expected values from a JSON file
        output_file_path = os.path.join(self.base_dir, 'expected_outputs', 'KorakianitisMixedModel_expected_output.json')

        # Verify the file exists
        self.assertTrue(os.path.exists(output_file_path), f"Expected output file not found: {output_file_path}")

        with open(output_file_path, 'r') as f:
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
        """
        Test the functionality of the solver by running it with different step sizes and verifying its behavior.
        This test ensures that:
        1. The solver can be configured and run with various step sizes.
        2. The solver converges successfully for each step size.
        3. The state variables of the model components are updated correctly after solving.
        4. The computed results match the expected values within a specified tolerance.

        """

        cycle_step_sizes = [1, 3, 5, 7]  # Define the step sizes to test

        for i_cycle_step_size in cycle_step_sizes:

            # Use logging to print the current step size
            logging.info(f"Testing solver with step size: {i_cycle_step_size}")

            with self.subTest(cycle_step_size=i_cycle_step_size):

                # Initializing the parameter object
                self.parobj = KorakianitisMixedModel_parameters()

                # Initializing the model
                self.model = KorakianitisMixedModel(time_setup_dict=self.time_setup_dict,
                                                    parobj=self.parobj,
                                                    suppress_printing=True)

                # Initializing the solver
                self.solver = Solver(model=self.model)

                # Reconfigure the solver with the current step size
                self.solver.setup(suppress_output=True, method='LSODA', step=i_cycle_step_size)

                # Running the model
                self.solver.solve()

                # Verify the solver converged
                self.assertTrue(self.solver.converged or self.solver._Nconv is not None)

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
                expected_ndarray = np.array(
                    [self.expected_values["results"][str(i_cycle_step_size)][key1][key2] for key1 in new_dict.keys() for key2 in new_dict[key1].keys()]
                    )
                new_ndarray = np.array([new_dict[key1][key2] for key1 in new_dict.keys() for key2 in new_dict[key1].keys()])
                test_ndarray = np.where(np.abs(expected_ndarray) > 1e-6,
                                        np.abs((expected_ndarray - new_ndarray) / expected_ndarray),
                                        np.abs((expected_ndarray - new_ndarray)))
                self.assertTrue((test_ndarray < RELATIVE_TOLERANCE).all())


if __name__ == '__main__':
    unittest.main()
