import unittest
from CE4021.Calculus import derive_poly  # Replace with the actual import for your derive_poly function


class TestDerivePoly(unittest.TestCase):

    def test_derive_poly_descending(self):
        poly = [3, 2, 1]  # Represents 3x^2 + 2x + 1
        result = derive_poly(poly)
        expected = [6, 2]  # Derivative is 6x + 2
        self.assertEqual(result, expected)

    def test_derive_poly_ascending(self):
        poly = [1, 2, 3]  # Represents 3x^2 + 2x + 1
        result = derive_poly(poly, increasing=True)
        expected = [2, 6]  # Derivative is 6x + 2
        self.assertEqual(result, expected)

    def test_derive_poly_negative_coefficients_descending(self):
        poly = [-3, -2, -1]  # Represents -3x^2 - 2x - 1
        result = derive_poly(poly)
        expected = [-6, -2]  # Derivative is -6x - 2
        self.assertEqual(result, expected)

    def test_derive_poly_negative_coefficients_ascending(self):
        poly = [-1, -2, -3]  # Represents -3x^2 - 2x - 1
        result = derive_poly(poly, increasing=True)
        expected = [-2, -6]  # Derivative is -6x - 2
        self.assertEqual(result, expected)

    def test_derive_poly_zero_coefficients(self):
        poly = [1, 0, 0, 0]  # Represents x^3
        result = derive_poly(poly)
        expected = [3, 0, 0]  # Derivative is 3x^2
        self.assertEqual(result, expected)

    def test_derive_poly_constant(self):
        poly = [5]  # Represents a constant polynomial
        result = derive_poly(poly)
        expected = []  # Derivative is 0
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
