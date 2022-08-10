import numpy as np

from src_opt.allocator_future_demand import ResourceAllocatorFutureDemand
from src_opt.allocator_current_demand import ResourceAllocatorCurrentDemand


class Whatever:
    """

    Attributes
    ----
      K: number of different areas
      M: M-1 = maximum number of potential clients that can be predicted
    """

    def __init__(
            self,
            k_dim: int,
            m_dim: int,
            t_max: int,
            s: np.ndarray,
            access_mask: np.ndarray,
            cost_transport: np.ndarray,
            cost_waiting: np.ndarray
    ):
        self.K = k_dim
        self.M = m_dim
        self.T_max = t_max

        self.s = s.copy()
        self.access_mask = access_mask.copy()
        self.cost_transport = cost_transport.copy()
        self.cost_waiting = cost_waiting.copy()

        self.s_booked_all = np.zeros(self.K, self.T_max)

        self.alloc_current = ResourceAllocatorCurrentDemand(
            cost=self.cost_transport,
            k_dim=self.K,
            m_dim=self.M)

        self.alloc_future = ResourceAllocatorFutureDemand(
            access_mask=access_mask,
            cost_transport=cost_transport,
            cost_waiting=cost_waiting,
            k_dim=k_dim,
            m_dim=m_dim)

    def iteration_step(self, d: np.ndarray, d_future: np.ndarray):
        """ Single iteration step

        Resource allocation based on current demand
        Update states
        Resource allocation based on future demand
        Update states
        """

        a = self.alloc_current.resource_allocation(s=self.s, d=d)

        d_left = self._not_satisfied_demand(a=a, s=self.s, d=d)

        s_idle, s_booked = self._get_idle_and_booked_vehicles(
            a=a, s=self.s, d=d)

        a_future = self.alloc_future.resource_allocation(
            s=s_idle, d=d_future)

        # Available supply in the next time step
        s_released, s_booked_all = self._get_updated_booked_supply(
            s_booked=s_booked, s_booked_all=self.s_booked_all)

        self.s = a_future @ s_idle + s_released
        self.s_booked_all = s_booked_all

    def _get_updated_booked_supply(
            self, s_booked: np.ndarray, s_booked_all: np.ndarray
    ):
        """
        Update the information about the blocked vehicles with the vehicles that
        were just assigned to customer pickup requests.

        For simplicity we will assume that all newly booked vehicles will remain
        occupied for t_block = self.T_max time steps.

        Since we have no information (yet) about the customer travel destination
        we will pick it randomly.

        This means that after t_block time steps all newly booked vehicles will
        become available at random ares in the map.

        Args
        ----
          s_booked: shape = (K,)
              newly booked vehicles that have to be added to group of already
              occupied vehicles
          s_booked_all: shape = (K, T_max)
              s_blocked[k,t] = number of occupied vehicles that will become free
              after T_max time steps
        """

        # assert s_booked.ndim == 1
        # assert s_booked_all.ndim == 2
        # assert t_block > 0

        # total number of newly booked vehicles
        n_new = s_booked.sum()

        # pick random locations where they will be free after t_block steps
        destinations = np.random.choice(np.arange(self.K), size=(n_new,))

        # group destination counts by area
        new_entry = np.bincount(destinations)

        # supply that will be released in the nex iteration step
        s_released = s_booked_all[:, 0].copy()

        # shift all columns one step to the left
        s_booked_all_new = np.zeros_like(s_booked_all)
        s_booked_all_new[:, :-1] = s_booked_all[:, 1:].copy()
        s_booked_all_new[:, -1] = new_entry

        return s_released, s_booked_all_new

    def update_supply_with_newly_freed_vehicles(
            self,
            s: np.ndarray,
            s_blocked: np.ndarray
    ):
        """

        Args
        ----
          s:  shape = (K,); available supply
          s_blocked: shape = (K, T_max)
              s_booked_all[k,t] = number of occupied vehicles that will become free
              after T_max time steps
        """

        s = s.copy()
        s_blocked = s_blocked.copy()

        s += s_blocked[:, 0]
        s_blocked[:, 0] = 0

        return s, s_blocked

    def _get_idle_and_booked_vehicles(
            self,
            a: np.ndarray,
            s: np.ndarray,
            d: np.ndarray
    ):
        """

        Args
        ----
          s: shape (K,); suppply
          a: shape (K,K); action matrix
          d: shape (K,); demand vector
        """

        """
            a           = a_non_diag + a_diag
                        = a_non_diag + a_diag_booked + a_diag_idle
    
            d_booked    = np.minimum(d, a @ s)
                        = a_non_diag @ s + a_diag_booked * s
    
            d_booked - a_non_diag @ s = a_diag_booked * s
    
            a_diag_booked = (d_satisfied - a_non_diag @ s) / s
    
            a_diag_idle = a_diag - a_diag_booked
        """

        d_booked = np.minimum(d, a @ s)

        a_non_diag = a.copy()
        a_non_diag -= np.diag(a.diagonal())

        a_diag_booked = (d_booked - a_non_diag @ s) / s

        a_diag_idle = a.diagonal() - a_diag_booked

        s_idle = a_diag_idle * s
        s_booked = s - s_idle

        return s_idle, s_booked

    def _not_satisfied_demand(
            self,
            a: np.ndarray,
            s: np.ndarray,
            d: np.ndarray
    ):
        """ Current demand that wont be covered after assigning vehicles to
        customers """

        d_booked = np.minimum(d, a @ s)
        d_left = d - d_booked
        return d_left
