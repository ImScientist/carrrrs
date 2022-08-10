import itertools
import numpy as np
from scipy.optimize import minimize, Bounds


class ResourceAllocatorCurrentDemand:
    """ Resource allocator to cover present demand

    Attributes
    ----------
      s: supply vector
      d: demand vector
      cost: cost_transport matrix; c[i,j] is the cost_transport to move from j to i
      actions:
      demands:
      supplies:
      __tol: tolerance for the constraints
    """

    def __init__(self, cost: np.ndarray, k_dim: int, m_dim: int):

        self.K = k_dim  # number of different areas
        self.M = m_dim  # M-1 = maximum number of potential clients that can be predicted
        self.cost = cost.copy()

        self.s = None
        self.d = None
        self.a = None

        self.supplies = []
        self.demands = []
        self.actions = []
        self.costs_transport = []

        self.__tol = 1e-3
        self.__max_attempts = 500
        self.__min_successful_attempts = 200

    def _allocation_mat(self, x: np.ndarray):
        """ Map logits to allocation matrix """

        x = x.reshape(self.K, self.K)
        a = np.exp(x) / np.exp(x).sum(axis=0, keepdims=True)
        return a

    def _costs_transport(self, x: np.ndarray):
        """ Empty vehicle transportation costs """

        a = self._allocation_mat(x)

        # if there are no vehicles in a given area i (s[i] = 0)
        # We clip that value to 1 to force the i-th column of a that has no
        # impact on the final vehicle distribution to suggest no movement

        s = self.s.clip(1)

        return ((a * self.cost) @ s).sum()

    def _demand_constraint(self, x: np.ndarray):
        """ Demand constraint """

        a = self._allocation_mat(x)

        # number of requests that cannot be covered because demand > supply
        d_tot = self.d.sum()
        s_tot = self.s.sum()
        non_covered_requests = (d_tot - s_tot).clip(0)

        return ((self.d - a @ self.s).clip(0)).sum() - non_covered_requests

    def optimize(self):
        """
        TODO: transform into a convex optimization problem
        """

        # lower and upper bound for the logits
        lb = -10 * np.ones((self.K * self.K,))
        ub = 10 * np.ones((self.K * self.K,))

        params_scale = np.max([np.abs(ub), np.abs(lb)])

        x0 = np.random.uniform(
            low=lb / params_scale,
            high=ub / params_scale)

        bounds = Bounds(
            lb=lb / params_scale,
            ub=ub / params_scale)

        res = minimize(
            fun=lambda x: self._costs_transport(x * params_scale),
            x0=x0,
            bounds=bounds,
            constraints={
                'type': 'eq',
                'fun': lambda x: self._demand_constraint(x * params_scale)
            },
            options={'maxiter': 1000},
            tol=self.__tol)

        res.x *= params_scale
        res.cost_transport = self._costs_transport(res.x)

        demand_constraint = self._demand_constraint(res.x)

        if not -self.__tol <= demand_constraint <= self.__tol:
            res.success = False

        return res

    def resource_allocation(self, s: np.ndarray, d: np.ndarray):
        """ Find action `a` such that the supply meets the demand and the
        vehicle transportation costs are minimized """

        self.s = s.copy()
        self.d = d.copy()

        # Create iterator that stops after min_success converged optimisations
        results = (self.optimize() for _ in range(self.__max_attempts))
        results = filter(lambda r: r.success, results)
        results = itertools.islice(results, self.__min_successful_attempts)

        # Execute the iterator
        results = list(results)

        if len(results) < self.__min_successful_attempts:
            raise Exception(
                f'Optimization failed. Only {len(results)} out of'
                f' {self.__max_attempts} params searches were successful.')

        # Pick the result with the lowest transportation_cost
        res = min(results, key=lambda r: r.transportation_cost)

        self.a = self._allocation_mat(res.x)

        # Update lists...
        self.supplies.append(self.s.copy())
        self.demands.append(self.d.copy())
        self.actions.append(self.a.copy())
        self.costs_transport.append(res.cost_transport)

        return self.a.copy()
