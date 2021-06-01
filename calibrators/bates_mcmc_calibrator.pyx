"""
Largely a copy-clone of https://github.com/Imlerith/Jump-Diffusion-Calibrator,
this allows us to run a Markov Chain Monte Carlo calibration of a given asset returns
time-series to estimate the parameters needed to run our Stochastic Volatility Jump model (the Bates model).
"""
import numpy as np
cimport numpy as np
from scipy.stats import truncnorm, invgamma, binom, beta, norm
from tqdm import tqdm
cimport utils
from cpython cimport dict

def max_func(x):
    return max(0, x)

cdef class BatesCalibrator:
    # float alpha_lambda_prior = 2.0, float beta_lambda_prior = 40.0, float alpha_s_prior = 5.0,
    #             float beta_s_prior = 0.2, float mu_s = 0.0, float sigma_sq_s = 0.1
    cdef:
        double alpha_lambda_prior
        double beta_lambda_prior
        double alpha_s_star
        double beta_s_prior
        double mu_s
        double sigma_sq_s
        double mu_prior
        double sigma_sq_mu_prior
        double[:] returns
        double delta_t
        int T
        double s0
        double cost_of_carry
        double beta_prior
        double p_prior
        double psi_prior
        double theta_prior
        double sigma_sq_theta_prior
        double kappa_prior
        double sigma_sq_kappa_prior
        double alpha_star
        double _mu
        double _kappa
        double _theta
        double _omega
        double _psi
        double lambda_d

    def __init__(self, alpha_lambda_prior, beta_lambda_prior, alpha_s_prior,
                 beta_s_prior, mu_s, sigma_sq_s, mu_prior, sigma_sq_mu_prior, price_series, delta_t, cost_of_carry,
                 beta_prior, p_prior, psi_prior, theta_prior, sigma_sq_theta_prior, kappa_prior, sigma_sq_kappa_prior, alpha_prior, mu, kappa, theta, omega, psi):
        self.returns = np.array(np.diff(np.log(price_series)), dtype=np.float64)
        self.s0 = price_series[0]
        self.T = len(self.returns)
        # --- initialize prior parameters
        self.alpha_lambda_prior = alpha_lambda_prior
        self.beta_lambda_prior = beta_lambda_prior
        self.beta_s_prior = beta_s_prior
        self.alpha_s_star = self.T / 2 + alpha_s_prior
        # --- initialize posterior parameters
        self.mu_s = mu_s
        self.sigma_sq_s = sigma_sq_s
        self.lambda_d = beta.rvs(a=alpha_lambda_prior, b=beta_lambda_prior)
        # -- BasicCalibrator
        self.mu_prior = mu_prior
        self.sigma_sq_mu_prior = sigma_sq_mu_prior
        self.delta_t = delta_t
        self.cost_of_carry = cost_of_carry
        # --- HestonCalibrator
        self.beta_prior = beta_prior
        self.p_prior = p_prior
        self.psi_prior = psi_prior
        self.theta_prior = theta_prior
        self.sigma_sq_theta_prior = sigma_sq_theta_prior
        self.kappa_prior = kappa_prior
        self.sigma_sq_kappa_prior = sigma_sq_kappa_prior
        self.alpha_star = self.T / 2 + alpha_prior
        self._mu = mu
        self._kappa = kappa
        self._theta = theta
        self._omega = omega
        self._psi = psi

    # int n_mcmc_steps=10000, int burn_in=5000, float rejection_rate=0.005
    cpdef dict calibrate(self, int n_mcmc_steps, int burn_in, double rejection_rate):
        # ----- generate starting values for V using a truncated normal distribution
        #       (present-time as well as shifted backward and forward)
        V_t_array = np.array(truncnorm.rvs(a=0, b=np.inf, loc=0.0225, scale=0.005, size=self.T + 2))
        cdef np.ndarray[np.float64_t, ndim=1] V_t_minus_1_array = np.array(np.roll(V_t_array, 1), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] V_t_plus_1_array = np.array(np.roll(V_t_array, -1), dtype=np.float64)


        # ----- generate starting values for B using the Bernoulli distribution
        #       (present-time as well as shifted backward and forward)
        cdef np.ndarray[np.float64_t, ndim=1] B_t_array = np.array(binom.rvs(n=1, p=self.lambda_d, size=self.T + 2), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] B_t_plus_1_array = np.array(np.roll(B_t_array, -1), dtype=np.float64)
        B_t_plus_1_array[-1] = 0

        # ----- generate starting values for Z
        cdef np.ndarray[np.float64_t, ndim=1] Z_t_array = np.array(np.zeros((self.T + 2,)), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] Z_t_plus_1_array = np.array(np.roll(Z_t_array, -1), dtype=np.float64)
        Z_t_plus_1_array[-1] = 0

        # --- create a padded version of Y for computation purposes
        cdef np.ndarray[np.float64_t, ndim=1] Y_t_array = np.array(np.append(0, np.append(self.returns, 0)), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] Y_t_plus_1_array = np.array(np.roll(np.asarray(Y_t_array), -1), dtype=np.float64)
        Y_t_plus_1_array[-1] = 0

        num_params = 8
        _all_params_array_full = np.zeros((n_mcmc_steps, len(V_t_array) + len(B_t_array) + len(Z_t_array)
                                                + num_params))

        _all_params_array_full[0, 0:num_params] = np.array([self._mu, self._kappa, self._theta, self._psi,
                                                                 self._omega, self.mu_s, self.sigma_sq_s,
                                                                 self.lambda_d])
        _all_params_array_full[0, num_params:self.T + 2 + num_params] = V_t_array
        _all_params_array_full[0, self.T + 2 + num_params:2 * (self.T + 2) + num_params] = B_t_array
        _all_params_array_full[0, 2 * (self.T + 2) + num_params:3 * (self.T + 2) + num_params] = Z_t_array

        omega_alpha = self.alpha_star
        for itern in tqdm(range(1, n_mcmc_steps)):

            # ------- 1. Gibbs' sampling of model parameters -------
            # ----- start with the initialized parameters and update them using MCMC
            # (a) drift
            mu_mean = utils.mu_star(self._psi, self._omega, self._kappa, self._theta, V_t_array, self.returns,
                                   Z_t_array[1:-1], B_t_array[1:-1], self.delta_t, self.mu_prior,
                                   self.sigma_sq_mu_prior)
            mu_variance = utils.sigma_sq_star(self._psi, self._omega, V_t_array, self.delta_t, self.sigma_sq_mu_prior)
            self._mu = np.random.normal(mu_mean, np.sqrt(mu_variance))
            # (b) Omega
            omega_beta = utils.beta_star(V_t_array, self.returns, Z_t_array[1:-1], B_t_array[1:-1], self._mu, self.delta_t,
                                        self._kappa, self._theta, self.beta_prior, self.p_prior, self.psi_prior)
            self._omega = invgamma.rvs(omega_alpha, scale=omega_beta)
            # (c) psi
            psi_mean = utils.psi_star(self.returns, V_t_array, Z_t_array[1:-1], B_t_array[1:-1], self._mu, self.delta_t,
                                     self._kappa, self._theta, self.p_prior, self.psi_prior)
            psi_vola = np.sqrt(utils.sigma_sq_psi_star(self.returns, V_t_array, Z_t_array[1:-1], B_t_array[1:-1],
                                                      self._mu, self.delta_t, self.p_prior, self._omega))
            self._psi = np.random.normal(psi_mean, psi_vola)
            # (d) theta
            theta_mean = utils.theta_star(self.returns, V_t_array, Z_t_array[1:-1], B_t_array[1:-1], self._mu, self.delta_t,
                                         self._psi, self._kappa, self._omega, self.theta_prior,
                                         self.sigma_sq_theta_prior)
            theta_vola = np.sqrt(utils.sigma_sq_theta_star(V_t_array, self.delta_t, self._kappa,
                                                          self._omega, self.sigma_sq_theta_prior))
            self._theta = truncnorm.rvs((0 - theta_mean) / theta_vola, (5 - theta_mean) / theta_vola, loc=theta_mean,
                                        scale=theta_vola)
            # (e) kappa
            kappa_mean = utils.kappa_star(self.returns, V_t_array, Z_t_array[1:-1], B_t_array[1:-1], self._mu, self.delta_t,
                                         self._psi, self._theta, self._omega, self.kappa_prior,
                                         self.sigma_sq_kappa_prior)
            kappa_vola = np.sqrt(utils.sigma_sq_kappa_star(V_t_array, self.delta_t, self._theta,
                                                          self._omega, self.sigma_sq_kappa_prior))
            self._kappa = truncnorm.rvs((0 - kappa_mean) / kappa_vola, (5 - kappa_mean) / kappa_vola, loc=kappa_mean,
                                        scale=kappa_vola)
            # (f) mu_s: posterior mean of the jump size
            mean_m_s = utils.mu_m_s_star(self.s0, self.sigma_sq_s,
                                        self.T, Z_t_array[1:-1])  # posterior mean of the mean of the jump size
            var_m_s = utils.sigma_sq_m_s_star(self.s0, self.sigma_sq_s,
                                             self.T)  # posterior variance of the mean of the jump size
            self.mu_s = norm.rvs(mean_m_s, np.sqrt(var_m_s))
            # (g) sigma_sq_s: posterior variance of the jump size
            beta_s_star = self.beta_s_prior + 0.5 * sum(
                (Z_t_array[1:-1] - self.mu_s) ** 2)  # parameter of the posterior variance of the jump size
            self.sigma_sq_s = invgamma.rvs(a=self.alpha_s_star, scale=beta_s_star)
            # (h) lambda: the jump intensity
            alpha_lambda_star = self.alpha_lambda_prior + sum(B_t_array[1:-1])
            beta_lambda_star = self.beta_lambda_prior + self.T - sum(B_t_array[1:-1])
            self.lambda_d = beta.rvs(a=alpha_lambda_star, b=beta_lambda_star)
            # ------- 2. Metropolis-Hastings' sampling of variance paths -------
            Y_and_V_arrays = zip(Y_t_array, Y_t_plus_1_array, V_t_minus_1_array, V_t_array, V_t_plus_1_array,
                                 Z_t_array, Z_t_plus_1_array, B_t_array, B_t_plus_1_array)
            V_t_array_new = list()
            B_t_array_new = list()
            Z_t_array_new = list()
            for t, (Y_t, Y_t_plus_1, V_t_minus_1, V_t, V_t_plus_1, Z_t, Z_t_plus_1, B_t, B_t_plus_1) in enumerate(Y_and_V_arrays):

                # ----- simulate new jump size rvs Z
                # --- parameters of the posterior distribution of jump size
                mean_s_star = utils.mu_s_star(self._psi, self._omega, self._kappa, self._theta, V_t_minus_1, V_t, Y_t,
                                             self._mu, self.delta_t, self.mu_s, self.sigma_sq_s)
                var_s_star = utils.sigma_sq_s_star(self._psi, self._omega, V_t_minus_1, self.delta_t, self.sigma_sq_s)

                # --- simulate the jump size
                if B_t == 1:
                    # if a jump has happened, simulate the jump size with the first set of parameters
                    Z_t = norm.rvs(mean_s_star, np.sqrt(var_s_star))
                else:
                    # if no jump has happened, simulate the jump size with the second set of parameters
                    Z_t = norm.rvs(self.mu_s, np.sqrt(self.sigma_sq_s))

                # ----- simulate the Bernoulli rvs as jump indicators
                p_star = utils.get_p_star(self._psi, self._omega, self._kappa, self._theta, V_t_minus_1, V_t, Y_t, Z_t,
                                         self._mu, self.delta_t, self.lambda_d)
                B_t = binom.rvs(n=1, p=p_star)

                # ----- generate a proposal value
                V_proposal = np.random.normal(V_t, rejection_rate)

                # ----- get density of V at the previous and proposed values of V
                if t == 0:
                    V_density_at_curr = utils.state_space_target_dist_t_0(V_t, Y_t_plus_1, V_t_plus_1, Z_t_plus_1,
                                                                         B_t_plus_1, self.delta_t, self._mu,
                                                                         self._omega, self._psi, self._kappa,
                                                                         self._theta)
                    V_density_at_prop = utils.state_space_target_dist_t_0(V_proposal, Y_t_plus_1, V_t_plus_1, Z_t_plus_1,
                                                                         B_t_plus_1, self.delta_t, self._mu,
                                                                         self._omega, self._psi, self._kappa,
                                                                         self._theta)
                elif t != 0 and t <= len(self.returns):
                    V_density_at_curr = utils.state_space_target_dist_t_1_to_T(V_t, Y_t, Z_t, B_t, Y_t_plus_1,
                                                                              V_t_plus_1, V_t_minus_1, Z_t_plus_1,
                                                                              B_t_plus_1, self.delta_t, self._mu,
                                                                              self._omega, self._psi, self._kappa,
                                                                              self._theta)
                    V_density_at_prop = utils.state_space_target_dist_t_1_to_T(V_proposal, Y_t, Z_t, B_t, Y_t_plus_1,
                                                                              V_t_plus_1, V_t_minus_1, Z_t_plus_1,
                                                                              B_t_plus_1, self.delta_t, self._mu,
                                                                              self._omega, self._psi, self._kappa,
                                                                              self._theta)
                else:
                    V_density_at_curr = utils.state_space_target_dist_t_T_plus_1(V_t, Y_t, Z_t, B_t, V_t_minus_1,
                                                                                self.delta_t, self._mu, self._omega,
                                                                                self._psi, self._kappa, self._theta)
                    V_density_at_prop = utils.state_space_target_dist_t_T_plus_1(V_proposal, Y_t, Z_t, B_t, V_t_minus_1,
                                                                                self.delta_t, self._mu, self._omega,
                                                                                self._psi, self._kappa, self._theta)

                # ----- estimate an acceptance probability for a given variance value
                # corr_factor = norm.pdf(V_t, loc=V_proposal, scale=sigma_N) / norm.pdf(V_proposal, loc=V_t, scale=sigma_N)
                accept_prob = min(V_density_at_prop / V_density_at_curr, 1)
                u = np.random.uniform(0, 1)
                if u < accept_prob:
                    V_t = V_proposal

                # ----- update the state variables
                V_t_array_new.append(V_t)
                B_t_array_new.append(B_t)
                Z_t_array_new.append(Z_t)

            # ----- save the updated values
            # ----- get the next-step state variables
            V_t_array = np.array(V_t_array_new)
            B_t_array = np.array(B_t_array_new, dtype=np.float64)
            Z_t_array = np.array(Z_t_array_new, dtype=np.float64)

            V_t_minus_1_array = np.roll(V_t_array, 1)
            V_t_plus_1_array = np.roll(V_t_array, -1)
            B_t_plus_1_array = np.roll(B_t_array, -1)
            Z_t_plus_1_array = np.roll(Z_t_array, -1)
            _all_params_array_full[itern, 0:num_params] = np.array(
                [self._mu, self._kappa, self._theta, self._psi, self._omega, self.mu_s, self.sigma_sq_s, self.lambda_d])
            _all_params_array_full[itern, num_params:self.T + 2 + num_params] = V_t_array_new
            _all_params_array_full[itern, self.T + 2 + num_params:2 * (self.T + 2) + num_params] = B_t_array_new
            _all_params_array_full[itern, 2 * (self.T + 2) + num_params:3 * (self.T + 2) + num_params] = \
                Z_t_array_new

        # ----- get average estimates of model parameters
        all_params_array_no_burnin = _all_params_array_full[burn_in:, :]
        mu_final = np.mean(all_params_array_no_burnin[:, 0])
        kappa_final = np.mean(all_params_array_no_burnin[:, 1])
        theta_final = np.mean(all_params_array_no_burnin[:, 2])
        psi_final = np.mean(all_params_array_no_burnin[:, 3])
        omega_final = np.mean(all_params_array_no_burnin[:, 4])
        mu_s_final = np.mean(all_params_array_no_burnin[:, 5])
        sigma_sq_s_final = np.mean(all_params_array_no_burnin[:, 6])
        lambda_d_final = np.mean(all_params_array_no_burnin[:, 7])

        rho_final = np.sqrt(1 / (1 + omega_final / (psi_final ** 2)))
        volvol_final = psi_final / rho_final
        if volvol_final < 0:
            rho_final = -rho_final
            volvol_final = psi_final / rho_final
        return {"mu_final": mu_final, "kappa_final": kappa_final, "theta_final": theta_final,
                             "volvol_final": volvol_final, "rho_final": rho_final, "mu_s_final": mu_s_final,
                             "sigma_sq_s_final": sigma_sq_s_final, "lambda_d_final": lambda_d_final}

