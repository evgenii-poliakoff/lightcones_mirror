import lightcones as lc

def test_get_time():

    n_sites = 100
    # on-site energies
    es = [1]*n_sites
    # hoppings
    hs = [0.05]*(n_sites-1)

    # time grid
    dt = 0.01
    nt = 20000

    # spread
    spread = lc.spread(es, hs, nt, dt)

    # rho_plus
    rho_plus = lc.rho_plus(spread, dt)

    # minimal light cone
    rtol = 10**(-4)
    ti_arrival, spread_min, U_min, rho_plus_min = lc.minimal_forward_frame(spread, rho_plus, dt, rtol)

    # causal diamond dimension
    m = 4

    for m_in in range(1, len(ti_arrival)):
        ti = lc.get_incoming_time(ti_arrival, m_in)
        m_in_actual_prev = lc.m_in(ti_arrival, ti - 1)
        m_in_actual = lc.m_in(ti_arrival, ti)
        assert m_in_actual == m_in
        assert m_in_actual_prev == m_in - 1

    for m_out in range(1, len(ti_arrival) - m):
        ti = lc.get_outgoing_time(ti_arrival, m_out, m)
        m_out_actual_prev, _ = lc.get_inout_range(ti_arrival, ti - 1, m)
        m_out_actual, _ = lc.get_inout_range(ti_arrival, ti, m)
        assert m_out_actual == m_out
        assert m_out_actual_prev == m_out - 1
