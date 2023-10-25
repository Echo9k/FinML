import unittest
from CE4021 import Calculus
from fractions import Fraction


class TestPolynomials(unittest.TestCase):

    def setUp(self, function=Calculus.evaluate_polynomial):
        self.function = function

    def test_linear_evaluation(self):
        result = self.function(5, 3, 2)
        expected = 3 * 5 + 2
        self.assertEqual(result, expected)

    def test_quadratic_evaluation(self):
        result = self.function(-3, 2, -1, 4)
        expected = 2 * (-3) ** 2 - (-3) + 4
        self.assertEqual(result, expected)

    def test_cubic_evaluation(self):
        result = self.function(2.5, 1, 0, -3, 2)
        expected = 2.5 ** 3 - 3 * 2.5 + 2
        self.assertEqual(result, expected)

    def test_negative_coefficients_linear(self):
        result = self.function(5, -3, -2)
        expected = -3 * 5 - 2
        self.assertEqual(result, expected)

    def test_negative_coefficients_quadratic(self):
        result = self.function(-3, -2, 1, -4)
        expected = -2 * (-3) ** 2 + (-3) - 4
        self.assertEqual(result, expected)

    def test_fraction_coefficients_cubic(self):
        result = self.function(Fraction(5, 2), 1, 0, Fraction(-3, 2), Fraction(2, 3))
        expected = Fraction(5, 2) ** 3 - Fraction(3, 2) * Fraction(5, 2) + Fraction(2, 3)
        self.assertEqual(result, expected)

    def test_real_numbers_linear(self):
        result = self.function(4.5, 1.5, -2.5)
        expected = 1.5 * 4.5 - 2.5
        self.assertEqual(result, expected)

    def test_real_numbers_quadratic(self):
        result = self.function(-3.4, 2.3, -1.2, 4.7)
        expected = 2.3 * (-3.4) ** 2 - 1.2 * (-3.4) + 4.7
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
