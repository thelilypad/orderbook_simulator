cpdef double mu_star(double psi, double omega, double kappa, double theta, double[:] V, double[:] Y, double[:] Z,
              double[:] B, double dt, double mu_prior, double sigma_sq_mu_prior)

cpdef double sigma_sq_star(double psi, double omega, double[:] V, double dt, double sigma_prior)

cpdef double[:] get_eps_s(double[:] V, double[:] Y, double[:] Z, double[:] B, double mu, double dt)

cpdef double[:] get_eps_v(double[:] V, double dt, double kappa, double theta)

cpdef double beta_star(double[:] V, double[:] Y, double[:] Z, double[:] B, double mu, double dt, double kappa,
                double theta, double beta_prior, double p_prior, double psi_prior)

cpdef double psi_star(double[:] Y, double[:] V, double[:] Z, double[:] B, double mu, double dt, double kappa,
               double theta, double p_prior, double psi_prior)

cpdef double sigma_sq_psi_star(double[:] Y, double[:] V, double[:] Z, double[:] B, double mu, double dt,
                        double p_prior, double omega)

cpdef double theta_star(double[:] Y, double[:] V, double[:] Z, double[:] B, double mu, double dt, double psi,
                 double kappa, double omega, double theta_prior, double sigma_sq_theta_prior)

cpdef double sigma_sq_theta_star(double[:] V, double dt, double kappa, double omega, double sigma_sq_theta_prior)

cpdef double kappa_star(double[:] Y, double[:] V, double[:] Z, double[:] B, double mu, double dt, double psi,
                 double theta, double omega, double kappa_prior, double sigma_sq_kappa_prior)

cpdef double sigma_sq_kappa_star(double[:] V, double dt, double theta, double omega, double sigma_sq_kappa_prior)

cpdef double mu_s_star(double psi, double omega, double kappa, double theta, double V_t_minus_1, double V_t, double Y_t,
                double mu, double dt, double mu_s, double sigma_sq_s)

cpdef double sigma_sq_s_star(double psi, double omega, double V_t_minus_1, double dt, double sigma_sq_s)

cpdef double mu_m_s_star(double S_0, double sigma_sq_s, int T, double[:] Z)

cpdef double sigma_sq_m_s_star(double S_0, double sigma_sq_s, int T)

cpdef double get_p_star(double psi, double omega, double kappa, double theta, double V_t_minus_1, double V_t,
                 double Y_t, double Z_t, double mu_drift, double delta_t, double lambda_d)

cpdef double state_space_target_dist_term_1(double V_proposed_or_current, double Y_t_plus_1, double Z_t_plus_1,
                                     double B_t_plus_1, double dt, double mu, double omega, double psi)

cpdef double state_space_target_dist_term_2(double V_proposed_or_current, double Y_t_plus_1, double V_t_plus_1,
                                     double Z_t_plus_1, double B_t_plus_1, double dt, double mu, double omega,
                                     double psi, double kappa, double theta)

cpdef double state_space_target_dist_term_3(double V_proposed_or_current, double V_t_plus_1, double dt, double omega,
                                     double kappa, double theta)

cpdef double state_space_target_dist_term_4(double V_proposed_or_current, double Y_t, double Z_t, double B_t,
                                     double V_t_minus_1, double dt, double mu, double omega, double psi,
                                     double kappa, double theta)

cpdef double state_space_target_dist_term_5(double V_proposed_or_current, double V_t_minus_1, double dt, double omega,
                                     double kappa, double theta)

cpdef double state_space_target_dist_t_0(double V_proposed_or_current, double Y_t_plus_1, double V_t_plus_1,
                                  double Z_t_plus_1, double B_t_plus_1, double dt, double mu, double omega,
                                  double psi, double kappa, double theta)

cpdef double state_space_target_dist_t_1_to_T(double V_proposed_or_current, double Y_t, double Z_t, double B_t,
                                       double Y_t_plus_1, double V_t_plus_1, double V_t_minus_1, double Z_t_plus_1,
                                       double B_t_plus_1, double dt, double mu, double omega, double psi,
                                       double kappa, double theta)

cpdef double state_space_target_dist_t_T_plus_1(double V_proposed_or_current, double Y_t, double Z_t, double B_t,
                                         double V_t_minus_1, dt, double mu, double omega, double psi,
                                         double kappa, double theta)
