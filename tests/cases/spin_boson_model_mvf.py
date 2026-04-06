# test the moving frame

def compute_s_z_av():
    
    # import the required packages

    import numpy as np
    from lightcones.linalg import mv
    import lightcones as lc
    from lightcones import models
    from lightcones.solvers.schrodinger import solve
    
    # construct the moving frame:
    
    # chain
    n_sites = 100
    # on-site energies
    es = [1]*n_sites
    # hoppings
    hs = [0.05]*(n_sites-1)

    # time grid
    dt = 0.01
    nt = 10000

    # spread
    spread = lc.metric.spread(es, hs, nt, dt)

    # rho_plus
    rho_plus = lc.metric.rho_plus(spread, dt)

    # minimal forward light cone
    rtol = 10**(-4)
    ti_arrival, spread_min, U_min, rho_plus_min = lc.frames.minimal_forward(spread, rho_plus, dt, rtol)
    
    # causal diamond frame
    cd_dim = 4
    spread_cd, U_cd = lc.frames.causal_diamonds.fixed(spread_min, ti_arrival, U_min, rho_plus_min, dt, rtol, cd_dim)
    
    # moving frame
    spread_mv, H_mv = lc.frames.causal_diamonds.moving(spread_cd, ti_arrival, U_cd, dt, cd_dim)
    
    # Now solve the spin boson model in the moving frame
    num_modes = 15
    max_num_quanta = 5
    m = models.spin_boson(num_modes, max_num_quanta)

    wq = 1.0
    Hs = wq * m.s_p @ m.s_m
    g = 0.05
    V = g * m.s_m
    V_dag = g * m.s_p

    # Time step
    dt = 0.01

    # Final grid
    t_max = 100
    t = np.arange(0, t_max + dt, dt)
    n_time = t.size

    def f(ti):
        return(0.1*np.cos((ti + 0.5)*dt))

    psi_0 = np.zeros(m.dimension, dtype = complex)
    psi_0[0] = 1

    compute_s_z_av.Ht = None
    compute_s_z_av.Hint = m.space.zero_op

    def begin_step(ti, psi):
        m_out, m_in = lc.frames.causal_diamonds.get_inout_range_fixed(ti_arrival, ti, cd_dim)
        if m_in > 0:
            compute_s_z_av.Hint = V_dag @ sum(spread_mv[m_out : m_in, ti] * m.a[m_out : m_in])
            compute_s_z_av.Hint = compute_s_z_av.Hint + compute_s_z_av.Hint.conj().transpose()
        compute_s_z_av.Ht = Hs + m.s_x * f(ti) + compute_s_z_av.Hint
        Hw = m.zero_op
        W = lc.frames.causal_diamonds.get_H(ti_arrival, H_mv, ti)
        if not W is None:
            for p in range(m_in - m_out):
                for q in range(m_in - m_out):
                    Hw += 1j * m.a_dag[m_out + q] @ m.a[m_out + p] * W[q, p].conj()
        compute_s_z_av.Ht = compute_s_z_av.Ht + Hw
    
    def apply_h(ti, psi_in, psi_out):
        mv(compute_s_z_av.Ht, psi_in, psi_out)

    # Here we store the average of observables
    s_z_av = []

    def eval_o(ti, psi):
        s_z_av.append(np.vdot(psi, m.s_z @ psi))

    solve(0, n_time-1, dt, apply_h, psi_0, begin_step = begin_step, eval_o = eval_o)
    
    s_z_av = np.array(s_z_av)
    s_z_av_table = np.column_stack((t, s_z_av.real))
    
    return s_z_av_table