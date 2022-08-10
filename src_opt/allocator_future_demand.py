import itertools
import numpy as np
from scipy.optimize import minimize, Bounds


class ResourceAllocatorFutureDemand:
    """ Resource allocator to cover future demand

    Attributes
    ----
      K: number of different areas
      M: M-1 = maximum number of potential clients that can be predicted
      d: predicted demand:
          d[k, m] = probability to receive m new pickup orders in area k
      s: idle supply;
          s[k] = vehicles in area k that have not received an order
      access_mask: mask that denotes the locations that can be reached
          within one time step;
          access_mask[k,k'] = 1/0: the trip from k to k' is / is not possible
      cost_transport: cost matrix;
          c[k,k'] = cost to move a vehicle from k to k'
      cost_waiting: waiting time cost for a single customer;
          wt_cost[k] = waiting time cost for a customer in area k
      supplies: record idle supply from previous time steps
      demands: record predicted demands from previous time steps
      actions: record taken actions from previous time steps
      costs_transport:
      costs_waiting:
      costs_tot:
    """

    def __init__(
            self,
            access_mask: np.ndarray,
            cost_transport: np.ndarray,
            cost_waiting: np.ndarray,
            k_dim: int,
            m_dim: int
    ):
        self.K = k_dim  # number of different areas
        self.M = m_dim  # M-1 = maximum number of potential clients that can be predicted

        self.access_mask = access_mask.copy()
        self.cost_transport = cost_transport.copy()
        self.cost_waiting = cost_waiting.copy()

        self.s = None
        self.d = None
        self.a = None

        # record past problems and best actions
        self.supplies = []
        self.demands = []
        self.actions = []
        self.costs_transport = []
        self.costs_waiting = []
        self.costs_tot = []

        self.__tol = 1e-3
        self.__max_attempts = 500
        self.__min_successful_attempts = 200

    def _constrained_allocation_mat(self, x: np.ndarray):
        """ Map logits to allocation matrix by taking into account the
        access mask """

        x = x.reshape(self.K, self.K)
        x = np.exp(x) * self.access_mask
        a = x / x.sum(axis=0, keepdims=True)

        return a

    def _costs_waiting(self, x: np.ndarray):
        """ Customer waiting time costs """

        a = self._constrained_allocation_mat(x)

        # number of predicted clients shape = (1, M)
        m = np.arange(self.M).reshape(1, -1)

        # costs only if the number of predicted clients is higher than the
        # allocated supply
        # m: (1, M)
        # alloc_supply: (K, 1)
        # non_covered_demand: (K, M)
        # non_covered_demand[k,m] = difference btw the demanded (m) and
        #   allocated vehicles in area k
        s_alloc = a @ self.s.reshape(-1, 1)
        non_covered_demand = (m - s_alloc).clip(0)

        # cost_waiting: (K,) -> (K, 1)
        # non_covered_demand: (K, M)
        wt_cost_matrix = non_covered_demand * self.cost_waiting.reshape(-1, 1)

        # weight waiting cost_transport matrix with predicted demand d
        # d[k, m] = probability to get m customers in area k
        wt_costs = (wt_cost_matrix * self.d).sum()

        return wt_costs

    def _costs_transport(self, x: np.ndarray):
        """ Empty vehicle transportation costs """

        a = self._constrained_allocation_mat(x)

        # If there are no vehicles in a given area i (s[i] = 0)
        # We clip that value to 1 to force the i-th column of a that has no
        # impact on the final vehicle distribution to suggest no movement

        # cost_transport: (K, K)
        # a: (K, K)
        # s: (K,)
        transport_costs = ((self.cost_transport * a) @ self.s.clip(1)).sum()

        return transport_costs

    def _costs_tot(self, x: np.ndarray):
        return self._costs_waiting(x) + self._costs_transport(x)

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
            fun=lambda x: self._costs_tot(x * params_scale),
            x0=x0,
            bounds=bounds,
            options={'maxiter': 1000},
            tol=self.__tol)

        res.x *= params_scale
        res.cost_tot = self._costs_tot(res.x)
        res.cost_transport = self._costs_transport(res.x)
        res.cost_waiting = self._costs_waiting(res.x)

        return res

    def resource_allocation(self, s: np.ndarray, d: np.ndarray):
        """ Find action `a` such that the supply meets the predicted future
        demand such that there is a balance between the extra vehicle
        transportation costs and the decreased customer waiting time """

        # assert d.ndim == 2  # (K, M) with m-1 = max number of customers
        # assert s.ndim == 1  # (K,)
        # assert cost_transport.ndim == 2  # (K, M)
        # assert cost_waiting.ndim == 1  # (K,)

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

        # Pick the result with the lowest total costs
        res = min(results, key=lambda r: r.costs_tot)

        self.a = self._constrained_allocation_mat(res.x)

        # Update lists...
        self.supplies.append(self.s.copy())
        self.demands.append(self.d.copy())
        self.actions.append(self.a.copy())

        self.costs_transport.append(res.cost_transport)
        self.costs_waiting.append(res.cost_waiting)
        self.costs_tot.append(res.cost_tot)

        return self.a.copy()
