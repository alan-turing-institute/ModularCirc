import unittest
from ModularCirc.Models.NaghaviModel import NaghaviModelParameters, NaghaviModel
from ModularCirc.Solver import Solver

class TestModularCirc(unittest.TestCase):

    def setUp(self):
        self.time_setup_dict = {
            'name': 'TimeTest',
            'ncycles': 40,
            'tcycle': 1.0,
            'dt': 0.001,
            'export_min': 1
        }
        self.parobj = NaghaviModelParameters()
        self.model = NaghaviModel(time_setup_dict=self.time_setup_dict, parobj=self.parobj)
        self.solver = Solver(model=self.model)
        self.solver.setup()

    def test_model_initialization(self):
        self.assertIsInstance(self.model, NaghaviModel)
        self.assertEqual(self.model.time_setup_dict, self.time_setup_dict)
        self.assertEqual(self.model.parobj, self.parobj)

    def test_solver_initialization(self):
        self.assertIsInstance(self.solver, Solver)
        self.assertEqual(self.solver.model, self.model)

    def test_solver_run(self):
        self.solver.solve()
        self.assertTrue(hasattr(self.solver.model, 'components'))
        self.assertIn('lv', self.solver.model.components)
        self.assertTrue(len(self.solver.model.components['lv'].V.values) > 0)
        self.assertTrue(len(self.solver.model.components['lv'].P_i.values) > 0)

if __name__ == '__main__':
    unittest.main()
