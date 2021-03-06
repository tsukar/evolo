import unittest
from src.individual import Individual

class IndividualTestCase(unittest.TestCase):
    def test_load_and_save(self):
        Individual.load('cfg/seed.cfg', 0, 99).save()
        with open('individuals/00-99.cfg') as f:
            saved_individual = f.read()
        with open('cfg/ground-truth.cfg') as f:
            ground_truth = f.read()
        self.assertEqual(saved_individual, ground_truth)

if __name__ == '__main__':
    unittest.main()
