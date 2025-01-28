import unittest
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
        self.model = KorakianitisMixedModel(time_setup_dict=self.time_setup_dict, parobj=self.parobj)
        self.solver = Solver(model=self.model)
        self.solver.setup()

        self.initial_values = {}

        for key, value in self.model.commponents.items():
            self.initial_values[key] = {
                'V': value.V.values.mean(),
                'P_i': value.P_i.values.mean(),
                'Q_i': value.Q_i.values.mean()
            }

        # Expected values from a previous run
        self.expected_values = {'la': {'V': 86.29361864828056, 'P_i': 1.3463596745790851, 'Q_i': 90.59968155550254}, 'mi': {'V': 0.0, 'P_i': 1.3463596745790851, 'Q_i': 86.33390902640244}, 'lv': {'V': 59.138804423397644, 'P_i': 25.959881273118285, 'Q_i': 86.33390902640244}, 'ao': {'V': 0.0, 'P_i': 25.959881273118285, 'Q_i': 78.75019226787782}, 'sas': {'V': 9.818789889013463, 'P_i': 122.7348736126631, 'Q_i': 78.75019226787782}, 'sat': {'V': 195.39715170393742, 'P_i': 122.12321981496237, 'Q_i': 117.98734314176914}, 'svn': {'V': 159.79374319303628, 'P_i': 7.794816741123734, 'Q_i': 106.83895350857598}, 'ra': {'V': 54.75886810642046, 'P_i': 0.7858211510595732, 'Q_i': 93.4532745341888}, 'ti': {'V': 0.0, 'P_i': 0.7858211510595732, 'Q_i': 91.42440301323052}, 'rv': {'V': 65.67855889569168, 'P_i': 11.108968588959739, 'Q_i': 91.42440301323052}, 'po': {'V': 0.0, 'P_i': 11.108968588959739, 'Q_i': 88.83083714042574}, 'pas': {'V': 5.941821072574982, 'P_i': 33.010117069866396, 'Q_i': 88.83083714042574}, 'pat': {'V': 124.43450990744847, 'P_i': 32.74592365985507, 'Q_i': 106.70569255158107}, 'pvn': {'V': 38.744134160198236, 'P_i': 1.8899577639121004, 'Q_i': 99.5089793327935}}


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

        new_dict = {}
        for key, value in self.model.commponents.items():

            new_dict[key] = {
                'V': value.V.values.mean(),
                'P_i': value.P_i.values.mean(),
                'Q_i': value.Q_i.values.mean()
            }

        # Check that the values have changed wrt the initial values
        self.assertFalse(self.initial_values == new_dict)

        # Check that the values are the same as the expected values
        self.assertTrue(new_dict == self.expected_values)



if __name__ == '__main__':
    unittest.main()
