import cvxpy as cvx
import numpy as np


class PortfolioConstraints:
    @staticmethod
    def long_onlys():
        return lambda weights: weights >= 0

    @staticmethod
    def leverage_limit(limit):
        return lambda weights: cvx.norm(weights, 1) <= limit

    @staticmethod
    def minimum_return(min_return, returns):
        return lambda weights: weights @ returns >= min_return
    
    @staticmethod
    def maximum_risk(max_risk, cov_matrix):
        return lambda weights: cvx.quad_form(weights, cov_matrix) <= max_risk
    
    @staticmethod
    def maximum_drawdown(max_drawdown, returns):
        return lambda weights: cvx.cumsum(weights @ returns) >= max_drawdown
    
    @staticmethod
    def max_stock_percentage(max_stock_percentage):
        return lambda weights: cvx.sum(weights) <= max_stock_percentage

def portfolio_variance(varA, varB, correlation):
    """
    Optimize a portfolio with two assets.

    Parameters
    ----------
    varA : float
        Variance of asset A.
    varB : float
        Variance of asset B.
    correlation : float
        Correlation between asset A and B.

    Returns
    -------
    optimal_weights : np.ndarray
        Optimal weights of asset A and B.
    minimized_variance : float
        Variance of the optimized portfolio.
    """
    x = cvx.Variable(2)
    cov_matrix = np.array([[varA, correlation * np.sqrt(varA * varB)],
                           [correlation * np.sqrt(varA * varB), varB]])

    objective = cvx.Minimize(cvx.quad_form(x, cov_matrix))
    constraints = [cvx.sum(x) == 1]
    problem = cvx.Problem(objective, constraints)
    problem.solve(qcp=True, solver=cvx.ECOS)

    if problem.status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
        raise ValueError("Optimization failed.")

    minimized_variance = problem.value
    optimal_weights = x.value

    return optimal_weights, minimized_variance


def portfolio_volatility(mean_returns, cov_matrix):
    """
    Calculates the optimal portfolio weights and minimized volatility using quadratic programming.

    Parameters:
    mean_returns (numpy.ndarray): Array of mean returns for each asset.
    cov_matrix (numpy.ndarray): Covariance matrix of asset returns.

    Returns:
    tuple: A tuple containing the optimal portfolio weights and the minimized volatility.
    """
    
    x = cvx.Variable(len(mean_returns))

    objective = cvx.Minimize(cvx.sqrt(cvx.quad_form(x, cov_matrix)))
    constraints = [cvx.sum(x) == 1, x >= 0]  # Including a long-only constraint
    problem = cvx.Problem(objective, constraints)
    problem.solve(qcp=True, solver=cvx.ECOS)

    if problem.status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
        raise ValueError("Optimization failed.")

    minimized_volatility = problem.value
    optimal_weights = x.value

    return optimal_weights, minimized_volatility


def optimize_twoasset_portfolio(varA, varB, correlation):
    """
    Optimize a portfolio with two assets.

    Parameters
    ----------
    varA : float
        Variance of asset A.
    varB : float
        Variance of asset B.
    correlation : float
        Correlation between asset A and B.

    Returns
    -------
    optimal_weights : np.ndarray
        Optimal weights of asset A and B.
    minimized_variance : float
        Variance of the optimized portfolio.
    """
    return portfolio_variance(varA, varB, correlation)

class PortfolioOptimization:
    """
    Class for portfolio optimization using cvxpy.
    """

    def __init__(self, cov_matrix, **kwargs):
        self.n_assets = cov_matrix.shape[0]
        self.cov_matrix = cov_matrix
        self.weights = cvx.Variable(self.n_assets)
        self.constraints = []
        self._add_default_constraints()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _add_default_constraints(self):
        # Assuming that weights are already initialized here
        self.constraints.append(PortfolioConstraints.long_onlys()(self.weights))
        self.constraints.append(PortfolioConstraints.leverage_limit(1)(self.weights))

    def _objective_risk_parity(self, risk_contributions_target):
        asset_risks = cvx.diag(self.cov_matrix) ** 0.5
        risk_contributions = cvx.multiply(asset_risks, self.weights)
        return cvx.Minimize(cvx.norm(risk_contributions - risk_contributions_target))

    def _objective_sharpe_ratio(self, returns, risk_free_rate):
        """
        Maximize Sharpe Ratio with a DCP compliant formulation.
        Squaring the Sharpe Ratio to maintain convexity.
        """
        expected_return = self.weights @ returns
        portfolio_variance = cvx.quad_form(self.weights, self.cov_matrix)

        # Use a risk aversion parameter to balance return and risk
        # The Sharpe Ratio is squared to maintain convexity
        sharpe_ratio = cvx.square((expected_return - risk_free_rate) / cvx.sqrt(portfolio_variance))
        return cvx.Maximize(sharpe_ratio - risk_free_rate * portfolio_variance)

    def _objective_target_return(self, returns, target_return):
            """
            Minimize variance subject to a target return constraint.

            Parameters:
            - returns (numpy.ndarray): Array of expected returns for each asset.
            - target_return (float): Target return value.

            Returns:
            - cvx.Minimize: Objective function to minimize portfolio variance.
            - list: List of return constraints.

            Raises:
            - ValueError: If the target return is not achievable with the provided expected returns.
            """
            portfolio_variance = cvx.quad_form(self.weights, self.cov_matrix)
            return_constraints = [self.weights @ returns >= target_return]

            # Ensure the constraints are feasible
            if not any(returns >= target_return):
                raise ValueError("Target return is not achievable with provided expected returns.")

            return cvx.Minimize(portfolio_variance), return_constraints

    def _objective_max_diversification(self):
        """
        Maximize diversification ratio with DCP compliant formulation.
        """
        asset_volatilities = cvx.diag(self.cov_matrix) ** 0.5
        diversification_measure = cvx.sum(cvx.multiply(asset_volatilities, self.weights))

        return cvx.Maximize(diversification_measure)

    def _objective_variance(self):
        return cvx.Minimize(cvx.quad_form(self.weights, self.cov_matrix))

    def _format_results(self, problem):
        return {
            'weights': np.array(self.weights.value).flatten(),
            'objective': problem.value
        }

    def optimize(self, objective='min_variance', *, custom_objective=None, **kwargs):
        """
        Optimize the portfolio weights based on the specified objective.

        Parameters:
            objective (str): The objective function to optimize. Default is 'min_variance'.
                Options include: 'min_variance', 'max_sharpe', 'risk_parity', 'target_return', 'max_diversification'.
            custom_objective (callable): A custom objective function that takes the current weights and additional
                keyword arguments as input and returns the objective function and additional constraints.
            **kwargs: Additional keyword arguments specific to each objective function.

        Returns:
            dict: A dictionary containing the optimized portfolio weights and other relevant information.
        """
        if custom_objective:
            objective_func, additional_constraints = custom_objective(self.weights, **kwargs)
            self.constraints.extend(additional_constraints)
        elif objective == 'min_variance':
            objective_func = self._objective_variance()
        elif objective == 'max_sharpe':
            objective_func = self._objective_sharpe_ratio(returns=kwargs["returns"],
                                                          risk_free_rate=kwargs['risk_free_rate'])
        elif objective == 'risk_parity':
            risk_contributions_target = kwargs.get('risk_contributions_target', np.ones(self.n_assets) / self.n_assets)
            objective_func = self._objective_risk_parity(risk_contributions_target)
        elif objective == 'target_return':
            target_return = kwargs['target_return']
            print(f"{target_return=:}")
            objective_func, return_constraints = self._objective_target_return(
                target_return=target_return,
                returns=kwargs['returns'])
            self.constraints.extend(return_constraints)
        elif objective == 'max_diversification':
            objective_func = self._objective_max_diversification()
        elif objective is None and custom_objective is None:
            raise ValueError("Must specify an objective function.")
        else:
            raise ValueError(f"Unrecognized objective: {objective}")

        problem = cvx.Problem(objective_func, self.constraints)
        problem.solve(qcp=True, solver=cvx.ECOS)

        if problem.status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
            raise ValueError(f"Optimization failed: {problem.status}")

        return self._format_results(problem)