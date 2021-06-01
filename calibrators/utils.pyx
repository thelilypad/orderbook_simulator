import numpy as np

cpdef double mu_star(double psi, double omega, double kappa, double theta, double[:] V, double[:] Y, double[:] Z,
              double[:] B, double dt, double mu_prior, double sigma_sq_mu_prior):
    numerator = sum((omega + psi ** 2) * (np.asarray(Y) + 0.5 * np.asarray(V[:-2]) * dt - \
                np.asarray(Z) * np.asarray(B)) / (omega * np.asarray(V[:-2]))) - \
                sum(psi * (np.asarray(V[1:-1]) - kappa * theta * dt - (1 - kappa * dt) * np.asarray(V[:-2])) /
                (omega * np.asarray(V[:-2]))) + mu_prior / sigma_sq_mu_prior
    denominator = dt * sum((omega + psi ** 2) / (omega * np.asarray(V[:-2]))) + 1 / sigma_sq_mu_prior
    return numerator / (denominator + 1e-08)

cpdef double sigma_sq_star(double psi, double omega, double[:] V, double dt, double sigma_prior):
    denominator = dt * sum((omega + psi ** 2) / (omega * np.asarray(V[:-2]))) + 1 / (sigma_prior ** 2)
    return 1 / (denominator + 1e-08)

cpdef double[:] get_eps_s(double[:] V, double[:] Y, double[:] Z, double[:] B, double mu, double dt):
    return (np.asarray(Y) - mu * dt + 0.5 * np.asarray(V[:-2]) * dt - np.asarray(Z) * np.asarray(B)) / \
           np.sqrt(np.asarray(V[:-2]) * dt)

cpdef double[:] get_eps_v(double[:] V, double dt, double kappa, double theta):
    return (np.asarray(V[1:-1]) - kappa * theta * dt - (1 - kappa * dt) * np.asarray(V[:-2])) / \
           np.sqrt(np.asarray(V[:-2]) * dt)

cpdef double beta_star(double[:] V, double[:] Y, double[:] Z, double[:] B, double mu, double dt, double kappa,
                double theta, double beta_prior, double p_prior, double psi_prior):
    eps_S = get_eps_s(V, Y, Z, B, mu, dt)
    eps_V = get_eps_v(V, dt, kappa, theta)
    result = beta_prior + 0.5 * sum(np.asarray(eps_V) ** 2) + 0.5 * p_prior * psi_prior ** 2 - \
             0.5 * ((p_prior * psi_prior + sum(np.asarray(eps_S) * np.asarray(eps_V))) ** 2 /
             (p_prior + sum(np.asarray(eps_S) ** 2)))
    return result

cpdef double psi_star(double[:] Y, double[:] V, double[:] Z, double[:] B, double mu, double dt, double kappa,
               double theta, double p_prior, double psi_prior):
    eps_S = get_eps_s(V, Y, Z, B, mu, dt)
    eps_V = get_eps_v(V, dt, kappa, theta)
    result = (p_prior * psi_prior + sum(np.asarray(eps_S) * np.asarray(eps_V))) / \
             (p_prior + sum(np.asarray(eps_S) ** 2))
    return result

cpdef double sigma_sq_psi_star(double[:] Y, double[:] V, double[:] Z, double[:] B, double mu, double dt,
                        double p_prior, double omega):
    eps_S = get_eps_s(V, Y, Z, B, mu, dt)
    result = omega / (p_prior + sum(np.asarray(eps_S) ** 2))
    return result

cpdef double theta_star(double[:] Y, double[:] V, double[:] Z, double[:] B, double mu, double dt, double psi,
                 double kappa, double omega, double theta_prior, double sigma_sq_theta_prior):
    numerator = sum(kappa * (np.asarray(V[1:-1]) - (1 - kappa * dt) * np.asarray(V[:-2])) / \
                (omega * np.asarray(V[:-2]))) - sum(psi * (np.asarray(Y) - mu * dt + 0.5 *
                np.asarray(V[:-2]) * dt - np.asarray(Z) * np.asarray(B)) * kappa / (omega * np.asarray(V[:-2])) +
                theta_prior / sigma_sq_theta_prior)
    denominator = dt * sum(kappa ** 2 / (omega * np.asarray(V[:-2]))) + 1 / sigma_sq_theta_prior
    return numerator / (denominator + 1e-08)

cpdef double sigma_sq_theta_star(double[:] V, double dt, double kappa, double omega, double sigma_sq_theta_prior):
    denominator = dt * sum(kappa ** 2 / (omega * np.asarray(V[:-2]))) + 1 / sigma_sq_theta_prior
    return 1 / (denominator + 1e-08)

cpdef double kappa_star(double[:] Y, double[:] V, double[:] Z, double[:] B, double mu, double dt, double psi,
                 double theta, double omega, double kappa_prior, double sigma_sq_kappa_prior):
    numerator = sum((theta - np.asarray(V[1:-1])) * (np.asarray(V[1:-1]) - np.asarray(V[:-2])) /
                (omega * np.asarray(V[:-2]))) - sum(psi * (np.asarray(Y) - mu * dt + 0.5 * np.asarray(V[:-2]) * dt -
                np.asarray(Z) * np.asarray(B)) * (theta - np.asarray(V[:-2])) / (omega * np.asarray(V[:-2]))) + \
                kappa_prior / sigma_sq_kappa_prior
    denominator = dt * sum((np.asarray(V[:-2]) - theta) ** 2 / (omega * np.asarray(V[:-2]))) + 1 / sigma_sq_kappa_prior
    return numerator / (denominator + 1e-08)

cpdef double sigma_sq_kappa_star(double[:] V, double dt, double theta, double omega, double sigma_sq_kappa_prior):
    denominator = dt * sum((np.asarray(V[:-2]) - theta) ** 2 / (omega * np.asarray(V[:-2]))) + 1 / sigma_sq_kappa_prior
    return 1 / (denominator + 1e-08)

cpdef double mu_s_star(double psi, double omega, double kappa, double theta, double V_t_minus_1, double V_t, double Y_t,
                double mu, double dt, double mu_s, double sigma_sq_s):
    numerator = ((omega + psi ** 2) * (Y_t + 0.5 * V_t_minus_1 * dt - mu * dt) / (omega * V_t_minus_1 * dt)) - \
                (psi * (V_t - kappa * theta * dt - (1 - kappa * dt) * V_t_minus_1) / (omega * V_t_minus_1 * dt)) \
                + mu_s / sigma_sq_s
    denominator = (omega + psi ** 2) / (omega * V_t_minus_1 * dt) + 1 / sigma_sq_s
    return numerator / (denominator + 1e-08)

cpdef double sigma_sq_s_star(double psi, double omega, double V_t_minus_1, double dt, double sigma_sq_s):
    denominator = (omega + psi ** 2) / (omega * V_t_minus_1 * dt) + 1 / sigma_sq_s
    return 1 / (denominator + 1e-08)

cpdef double mu_m_s_star(double S_0, double sigma_sq_s, int T, double[:] Z):
    numerator = sum(np.asarray(Z) / sigma_sq_s)
    denominator = 1 / S_0 + T / sigma_sq_s
    return numerator / (denominator + 1e-08)

cpdef double sigma_sq_m_s_star(double S_0, double sigma_sq_s, int T):
    denominator = 1 / S_0 + T / sigma_sq_s
    return 1 / (denominator + 1e-08)

cpdef double get_p_star(double psi, double omega, double kappa, double theta, double V_t_minus_1, double V_t,
                 double Y_t, double Z_t, double mu_drift, double delta_t, double lambda_d):
    A = ((omega + psi ** 2) * (
            Z_t ** 2 - 2 * Z_t * (Y_t - mu_drift * delta_t + 0.5 * V_t_minus_1 * delta_t)) + 2 * psi * (
                 V_t - kappa * theta * delta_t - (1 - kappa * delta_t) * V_t_minus_1) * Z_t) / (
                omega * V_t_minus_1 * delta_t)
    denominator = (1 - lambda_d) * np.exp(0.5 * A) / lambda_d + 1
    return 1 / (denominator + 1e-08)

cpdef double state_space_target_dist_term_1(double V_proposed_or_current, double Y_t_plus_1, double Z_t_plus_1,
                                     double B_t_plus_1, double dt, double mu, double omega, double psi):
    return (-1 / (2 * omega)) * (((omega + psi ** 2) * (
            0.5 * V_proposed_or_current * dt + Y_t_plus_1 - Z_t_plus_1 * B_t_plus_1 - mu * dt) ** 2) / (
                                         V_proposed_or_current * dt))

cpdef double state_space_target_dist_term_2(double V_proposed_or_current, double Y_t_plus_1, double V_t_plus_1,
                                     double Z_t_plus_1, double B_t_plus_1, double dt, double mu, double omega,
                                     double psi, double kappa, double theta):
    return (-1 / (2 * omega)) * (
            (-2 * psi * (0.5 * V_proposed_or_current * dt + Y_t_plus_1 - Z_t_plus_1 * B_t_plus_1 -
                         mu * dt) * (
                     (kappa * dt - 1) * V_proposed_or_current - kappa * theta * dt + V_t_plus_1)) / (
                    V_proposed_or_current * dt))

cpdef double state_space_target_dist_term_3(double V_proposed_or_current, double V_t_plus_1, double dt, double omega,
                                     double kappa, double theta):
    return (-1 / (2 * omega)) * (
            ((kappa * dt - 1) * V_proposed_or_current - kappa * theta * dt + V_t_plus_1) ** 2 / (
            V_proposed_or_current * dt))

cpdef double state_space_target_dist_term_4(double V_proposed_or_current, double Y_t, double Z_t, double B_t,
                                     double V_t_minus_1, double dt, double mu, double omega, double psi,
                                     double kappa, double theta):
    return (-1 / (2 * omega)) * (
            -2 * psi * (Y_t - Z_t * B_t - mu * dt + 0.5 * V_t_minus_1 * dt) * (V_proposed_or_current -
                                                                               kappa * theta * dt - (
                                                                                       1 - kappa * dt) * V_t_minus_1) / (
                    V_t_minus_1 * dt))

cpdef double state_space_target_dist_term_5(double V_proposed_or_current, double V_t_minus_1, double dt, double omega,
                                     double kappa, double theta):
    return (-1 / (2 * omega)) * (
            (V_proposed_or_current - kappa * theta * dt - (1 - kappa * dt) * V_t_minus_1) ** 2 / (V_t_minus_1 * dt))

cpdef double state_space_target_dist_t_0(double V_proposed_or_current, double Y_t_plus_1, double V_t_plus_1,
                                  double Z_t_plus_1, double B_t_plus_1, double dt, double mu, double omega,
                                  double psi, double kappa, double theta):
    multiplier = 1 / (V_proposed_or_current * dt)
    term_1 = state_space_target_dist_term_1(V_proposed_or_current, Y_t_plus_1, Z_t_plus_1, B_t_plus_1,
                                                dt, mu, omega, psi)
    term_2 = state_space_target_dist_term_2(V_proposed_or_current, Y_t_plus_1, V_t_plus_1, Z_t_plus_1,
                                                B_t_plus_1,
                                                dt, mu, omega, psi, kappa, theta)
    term_3 = state_space_target_dist_term_3(V_proposed_or_current, V_t_plus_1, dt, omega, kappa, theta)
    return multiplier * np.exp(term_1 + term_2 + term_3)

cpdef double state_space_target_dist_t_1_to_T(double V_proposed_or_current, double Y_t, double Z_t, double B_t,
                                       double Y_t_plus_1, double V_t_plus_1, double V_t_minus_1, double Z_t_plus_1,
                                       double B_t_plus_1, double dt, double mu, double omega, double psi,
                                       double kappa, double theta):
    multiplier = 1 / (V_proposed_or_current * dt)
    term_1 = state_space_target_dist_term_1(V_proposed_or_current, Y_t_plus_1, Z_t_plus_1, B_t_plus_1,
                                                dt, mu, omega, psi)
    term_2 = state_space_target_dist_term_2(V_proposed_or_current, Y_t_plus_1, V_t_plus_1, Z_t_plus_1,
                                                B_t_plus_1,
                                                dt, mu, omega, psi, kappa, theta)
    term_3 = state_space_target_dist_term_3(V_proposed_or_current, V_t_plus_1, dt, omega, kappa, theta)
    term_4 = state_space_target_dist_term_4(V_proposed_or_current, Y_t, Z_t, B_t, V_t_minus_1,
                                                dt, mu, omega, psi, kappa, theta)
    term_5 = state_space_target_dist_term_5(V_proposed_or_current, V_t_minus_1, dt, omega, kappa, theta)
    return multiplier * np.exp(term_1 + term_2 + term_3 + term_4 + term_5)

cpdef double state_space_target_dist_t_T_plus_1(double V_proposed_or_current, double Y_t, double Z_t, double B_t,
                                         double V_t_minus_1, dt, double mu, double omega, double psi,
                                         double kappa, double theta):
    multiplier = 1 / (V_proposed_or_current * dt)
    term_4 = state_space_target_dist_term_4(V_proposed_or_current, Y_t, Z_t, B_t, V_t_minus_1,
                                                dt, mu, omega, psi, kappa, theta)
    term_5 = state_space_target_dist_term_5(V_proposed_or_current, V_t_minus_1, dt, omega, kappa, theta)
    return multiplier * np.exp(term_4 + term_5)
