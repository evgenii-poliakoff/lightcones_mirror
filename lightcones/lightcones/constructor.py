import numpy as np
import lightcones as lc

class constructor:
    def __init__(self, e, h, tmax, dt = 0.01, ns = -1, rtol = 10**(-3), dim = 4, ns_guard = 3, boundary_tolerance = 10**(-9)):
        if not len(e) == len(h):
            raise Exception("One should have nh = ne, where nh = number of hoppings, ne = number of on-site energies")

        # number of chain sites
        if ns < 0:
            ns = len(e)
        else:
            e = e[:ns]
            h = h[:ns]

        self.ns = ns

        # time grid
        self.t = np.arange(0, tmax + dt, dt)
        self.nt = self.t.size

        self.coupling = h[0]
        h = h[1:]

        # spread
        self.spread = lc.spread(e, h, self.nt, dt)

        boundary_amplitude = np.amax(np.abs(self.spread[-ns_guard :, :]))
        if boundary_amplitude  > boundary_tolerance:
            raise Exception('Chain end reached: Increase the length of chain')

        # rho_plus
        self.rho_plus = lc.rho_plus(self.spread, dt)

        # minimal forward light cone
        self.ti_arrival, self.spread_min, self.U_min, self.rho_plus_min = lc.minimal_forward_frame(self.spread, self.rho_plus, dt, rtol)

        # causal diamond frame
        self.spread_cd, self.U_cd = lc.causal_diamond_frame(self.spread_min, self.ti_arrival, self.U_min, self.rho_plus_min, dt, rtol, dim)

        # moving frame
        self.spread_mv, self.H_mv = lc.moving_frame(self.spread_cd, self.ti_arrival, self.U_cd, dt, dim)

        # take into account the coupling
        self.spread_mv = self.spread_mv * self.coupling
