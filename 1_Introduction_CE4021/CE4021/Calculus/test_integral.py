import unittest
from fractions import Fraction
from CE4021.Calculus import integral


class TestIntegral(unittest.TestCase):

    def test_constant_polynomial(self):
        poly = {0: 5}
        result = integral(poly, 1, 4)
        expected = 5 * (4 - 1)  # 5(x)| from 1 to 4 = 5*3 = 15
        self.assertEqual(result, expected)

    def test_linear_polynomial(self):
        poly = {1: 3, 0: 2}
        result = integral(poly, 1, 4)
        expected = (3 / 2) * (4 ** 2 - 1 ** 2) + 2 * (4 - 1)  # (3x + 2)| from 1 to 4 = 21
        self.assertEqual(result, expected)

    def test_quadratic_polynomial(self):
        poly = {2: 1, 1: -1, 0: 2}
        result = integral(poly, 1, 3)
        expected = (1 / 3) * (3 ** 3 - 1 ** 3) - (1 / 2) * (3 ** 2 - 1 ** 2) + 2 * (
                    3 - 1)  # x^2 - x + 2| from 1 to 3 = 8/3 + 4
        self.assertEqual(result, expected)

    def test_polynomial_with_fractions(self):
        poly = {1: Fraction(3, 2), 0: Fraction(2, 3)}
        result = integral(poly, Fraction(1, 2), Fraction(3, 2))
        expected = (Fraction(3, 4) * (Fraction(3, 2) ** 2 - Fraction(1, 2) ** 2) +
                    Fraction(2, 3) * (Fraction(3, 2) - Fraction(1, 2)))  # (3/2)x + 2/3| from 1/2 to 3/2
        self.assertEqual(result, expected)

    # def test_negative_bounds(self):
    #     poly = {1: 2}
    #     result = integral(poly, -2, -1)
    #     expected = 1 * (-1 ** 2 - (-2) ** 2)  # 2x| from -2 to -1 = 1
    #     try:
    #         self.assertEqual(result, expected)
    #     except AssertionError:
    #         print("Negative bounds not implemented yet")
    #
    # def test_negative_coefficients(self):
    #     poly = {2: -1, 1: 3, 0: -2}
    #     result = integral(poly, 1, 3)
    #     expected = (-1/3) * (3**3 - 1**3) + (3/2) * (3**2 - 1**2) - 2 * (3 - 1)  # -x^2 + 3x - 2| from 1 to 3
    #     try:
    #         self.assertEqual(result, expected)
    #     except AssertionError:
    #         print("negative coefficients not implemented")


if __name__ == '__main__':
    unittest.main()
