import unittest
from test_eval_poly import TestPolynomials
from test_integral import TestIntegral
from test_derive_poly import TestDerivePoly


class CustomTextTestResult(unittest.TextTestResult):
    def getDescription(self, test):
        return str(test)

    def startTest(self, test):
        unittest.TextTestResult.startTest(self, test)
        print(f"Running: {self.getDescription(test)}")


class CustomTextTestRunner(unittest.TextTestRunner):
    resultclass = CustomTextTestResult

    def run(self, test):
        result = super().run(test)
        if result.wasSuccessful():
            print('✅ All tests passed successfully.')
        return result


if __name__ == '__main__':
    print("Test Polynomial derivation")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDerivePoly)
    CustomTextTestRunner().run(suite)

    print("Test Polynomials")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPolynomials)
    CustomTextTestRunner().run(suite)

    print("Test Integrals")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegral)
    CustomTextTestRunner().run(suite)
