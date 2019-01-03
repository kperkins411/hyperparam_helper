import unittest
from learning_rate_generators import CosignVals,LinearDecrease, LinearIncreaseVals, TriangularVals, ReverseTriangularVals

__project__ = 'Python_tutorials'

NUMB_EVEN_SAMPLES=10
NUMB_ODD_SAMPLES=9
MIN_VAL = 0.1
MAX_VAL = 1.0

class TBase(unittest.TestCase):
    def sample_run(self,vals, numb_samples):
        self.ls = vals.getVals(max_val=MAX_VAL, min_val=MIN_VAL, numb_iterations=numb_samples)
        self.assertEqual(numb_samples, len(self.ls))
        self.unique_test(numb_samples)

    def unique_test(self,numb_samples):
        raise NotImplementedError

class TestCosign(TBase):
    def test_getVals(self):
        self.sample_run(CosignVals(),NUMB_EVEN_SAMPLES)
        self.sample_run(CosignVals(),NUMB_ODD_SAMPLES)

    def unique_test(self,numb_samples):
        self.assertAlmostEqual(self.ls[0], MAX_VAL)
        self.assertAlmostEqual(self.ls[numb_samples - 1], MIN_VAL)

class TestLinear_Increase(TBase):
    def test_getVals(self):
        self.sample_run(LinearIncreaseVals(), NUMB_EVEN_SAMPLES)
        self.sample_run(LinearIncreaseVals(), NUMB_ODD_SAMPLES)

    def unique_test(self, numb_samples):
        self.assertEqual(self.ls[0], MIN_VAL)
        self.assertAlmostEqual(self.ls[numb_samples - 1], MAX_VAL)

class TestLinear_Decrease(TBase):
    def test_getVals(self):
        self.sample_run(LinearDecrease(), NUMB_EVEN_SAMPLES)
        self.sample_run(LinearDecrease(), NUMB_ODD_SAMPLES)

    def unique_test(self, numb_samples):
        self.assertAlmostEqual(self.ls[0], MAX_VAL)
        self.assertAlmostEqual(self.ls[numb_samples - 1], MIN_VAL)

class TestTriangular(TBase):
    def test_getVals(self):
        self.sample_run(TriangularVals(), NUMB_EVEN_SAMPLES)
        self.sample_run(TriangularVals(), NUMB_ODD_SAMPLES)

    def unique_test(self, numb_samples):
        # self.assertAlmostEqual(self.ls[0], MAX_VAL)
        # self.assertAlmostEqual(self.ls[numb_samples - 1], MIN_VAL)
        self.assertAlmostEqual(self.ls[0], MIN_VAL)  # verify both ends
        self.assertAlmostEqual(self.ls[numb_samples - 1], MIN_VAL)
        midval = numb_samples // 2 + numb_samples % 2 - 1
        self.assertAlmostEqual(self.ls[midval], MAX_VAL)  # verify middle

class TestReverseTriangular(TBase):
    def test_getVals(self):
        self.sample_run(ReverseTriangularVals(), NUMB_EVEN_SAMPLES)
        self.sample_run(ReverseTriangularVals(), NUMB_ODD_SAMPLES)

    def unique_test(self, numb_samples):
        # self.assertAlmostEqual(self.ls[0], MAX_VAL)
        # self.assertAlmostEqual(self.ls[numb_samples - 1], MIN_VAL)
        self.assertAlmostEqual(self.ls[0], MAX_VAL)  # verify both ends
        self.assertAlmostEqual(self.ls[numb_samples - 1], MAX_VAL)
        midval = numb_samples // 2 + numb_samples % 2 - 1
        self.assertAlmostEqual(self.ls[midval], MIN_VAL)  # verify middle

