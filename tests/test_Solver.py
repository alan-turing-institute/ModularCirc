import unittest
import numpy as np
from ModularCirc.Models.OdeModel import OdeModel
from ModularCirc.Solver import Solver
from ModularCirc.Models.KorakianitisMixedModel import KorakianitisMixedModel
from ModularCirc.Models.KorakianitisMixedModel_parameters import KorakianitisMixedModel_parameters

class TestSolver(unittest.TestCase):

    def setUp(self):
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

    def test_generate_dfdt_functions(self):
        # Setup the solver
        self.solver.setup(suppress_output=True, method='LSODA')
        # Verify the generated functions
        self.assertIsNotNone(self.solver.pv_dfdt_global)
        self.assertIsNotNone(self.solver.s_u_update)
        self.assertIsNotNone(self.solver.optimize)

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