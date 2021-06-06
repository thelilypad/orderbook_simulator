import numpy as np
cimport numpy as np
from scipy.stats import truncnorm, invgamma, binom, beta, norm
from tqdm import tqdm
cimport utils
from cpython cimport dict

def max_func(x):
    return max(0, x)

STANDARD_ARGS = {
    "alpha_lambda_prior": 2.0,
    "beta_lambda_prior": 40.0,
    "alpha_s_prior": 5.0,
    "beta_s_prior": 0.2,
    "mu_s": 0.0,
    "sigma_sq_s": 0.1,
    "mu_prior": 0.001,
    "sigma_sq_mu_prior": 1.000,
    "delta_t": 1.000,
    "alpha_prior": 2.000,
    "beta_prior": 0.005,
    "p_prior": 2.000,
    "psi_prior": 0.000,
    "theta_prior": 0.000,
    "sigma_sq_theta_prior": 1.000,
    "kappa_prior": 0.000,
    "sigma_sq_kappa_prior": 1.000,
    "mu": 0.050,
    "kappa": 0.500,
    "theta": 0.100,
    "omega": 0.100,
    "psi": 0.000,
}

cdef class BatesCalibratorParams:
    def __init__(self, double alpha_lambda_prior, double beta_lambda_prior, double beta_s_prior, double alpha_s_prior, double mu_s, double sigma_sq_s,
                 double mu_prior, double sigma_sq_mu_prior, double delta_t, double beta_prior, double p_prior, double psi_prior, double theta_prior,
                 double sigma_sq_theta_prior, double kappa_prior, double sigma_sq_kappa_prior, double alpha_prior, double mu, double kappa, double theta,
                 double omega, double psi):
        # --- initialize prior parameters
        self.alpha_lambda_prior = alpha_lambda_prior
        self.beta_lambda_prior = beta_lambda_prior
        self.beta_s_prior = beta_s_prior
        self.alpha_s_prior = alpha_s_prior
        # --- initialize posterior parameters
        self.mu_s = mu_s
        self.sigma_sq_s = sigma_sq_s
        # -- BasicCalibrator
        self.mu_prior = mu_prior
        self.sigma_sq_mu_prior = sigma_sq_mu_prior
        self.delta_t = delta_t
        # --- HestonCalibrator
        self.beta_prior = beta_prior
        self.p_prior = p_prior
        self.psi_prior = psi_prior
        self.theta_prior = theta_prior
        self.sigma_sq_theta_prior = sigma_sq_theta_prior
        self.kappa_prior = kappa_prior
        self.sigma_sq_kappa_prior = sigma_sq_kappa_prior
        self.alpha_prior = alpha_prior
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.omega = omega
        self.psi = psi

    @staticmethod
    def create_from_opts(dict optional_args):
        merged_args = {**optional_args, **STANDARD_ARGS}
        return BatesCalibratorParams(
            alpha_lambda_prior = merged_args['alpha_lambda_prior'],
            beta_lambda_prior = merged_args['beta_lambda_prior'],
            beta_s_prior = merged_args['beta_s_prior'],
            alpha_s_prior = merged_args['alpha_s_prior'],
            mu_s = merged_args['mu_s'],
            sigma_sq_s = merged_args['sigma_sq_s'],
            mu_prior = merged_args['mu_prior'],
            sigma_sq_mu_prior = merged_args['sigma_sq_mu_prior'],
            delta_t = merged_args['delta_t'],
            beta_prior = merged_args['beta_prior'],
            p_prior = merged_args['p_prior'],
            psi_prior = merged_args['psi_prior'],
            theta_prior=merged_args['theta_prior'],
            sigma_sq_theta_prior=merged_args['sigma_sq_theta_prior'],
            kappa_prior=merged_args['kappa_prior'],
            sigma_sq_kappa_prior=merged_args['sigma_sq_kappa_prior'],
            alpha_prior=merged_args['alpha_prior'],
            mu=merged_args['mu'],
            kappa=merged_args['kappa'],
            theta=merged_args['theta'],
            omega=merged_args['omega'],
            psi=merged_args['psi']
        )


"""
Calibrator designed to compute parameters needed for running Bates (Stochastic Volatility with Jumps)
option pricing model. This is largely based on the method described in https://www.tandfonline.com/doi/full/10.1080/00949655.2014.926899
and uses a Markov Chain Monte Carlo (MCMC) approach to compute the parameters directly from historical stock returns
rather than option-based approaches.
Source code largely based on https://github.com/Imlerith/Jump-Diffusion-Calibrator/
"""

cdef class BatesCalibrator:
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

    """
    :param price_series - The stock returns data to train the calibrator on.
    :param cost_of_carry - The cost of carry
    :param bates_model_params - An object largely containing initial conditions for the calibrator to iterate against.
    """

    def __init__(self, double[:] price_series, double cost_of_carry, BatesCalibratorParams bates_model_params):
        self.returns = np.array(np.diff(np.log(price_series)), dtype=np.float64)
        self.s0 = price_series[0]
        self.T = len(self.returns)
        # --- initialize prior parameters
        self.alpha_lambda_prior = bates_model_params.alpha_lambda_prior
        self.beta_lambda_prior = bates_model_params.beta_lambda_prior
        self.beta_s_prior = bates_model_params.beta_s_prior
        self.alpha_s_star = self.T / 2 + bates_model_params.alpha_s_prior
        # --- initialize posterior parameters
        self.mu_s = bates_model_params.mu_s
        self.sigma_sq_s = bates_model_params.sigma_sq_s
        self.lambda_d = beta.rvs(a=bates_model_params.alpha_lambda_prior, b=bates_model_params.beta_lambda_prior)
        # -- BasicCalibrator
        self.mu_prior = bates_model_params.mu_prior
        self.sigma_sq_mu_prior = bates_model_params.sigma_sq_mu_prior
        self.delta_t = bates_model_params.delta_t
        self.cost_of_carry = cost_of_carry
        # --- HestonCalibrator
        self.beta_prior = bates_model_params.beta_prior
        self.p_prior = bates_model_params.p_prior
        self.psi_prior = bates_model_params.psi_prior
        self.theta_prior = bates_model_params.theta_prior
        self.sigma_sq_theta_prior = bates_model_params.sigma_sq_theta_prior
        self.kappa_prior = bates_model_params.kappa_prior
        self.sigma_sq_kappa_prior = bates_model_params.sigma_sq_kappa_prior
        self.alpha_star = self.T / 2 + bates_model_params.alpha_prior
        self._mu = bates_model_params.mu
        self._kappa = bates_model_params.kappa
        self._theta = bates_model_params.theta
        self._omega = bates_model_params.omega
        self._psi = bates_model_params.psi

    cdef resample_drift(self, np.ndarray[np.float64_t, ndim=1] Z_t_array, np.ndarray[np.float64_t, ndim=1] B_t_array, np.ndarray[np.float64_t, ndim=1] V_t_array):
        mu_mean = utils.mu_star(self._psi, self._omega, self._kappa, self._theta, V_t_array, self.returns,
                                Z_t_array[1:-1], B_t_array[1:-1], self.delta_t, self.mu_prior,
                                self.sigma_sq_mu_prior)
        mu_variance = utils.sigma_sq_star(self._psi, self._omega, V_t_array, self.delta_t, self.sigma_sq_mu_prior)
        return np.random.normal(mu_mean, np.sqrt(mu_variance))

    cdef resample_omega(self, np.ndarray[np.float64_t, ndim=1] Z_t_array, np.ndarray[np.float64_t, ndim=1] B_t_array, np.ndarray[np.float64_t, ndim=1] V_t_array):
        omega_beta = utils.beta_star(V_t_array, self.returns, Z_t_array[1:-1], B_t_array[1:-1], self._mu, self.delta_t,
                                     self._kappa, self._theta, self.beta_prior, self.p_prior, self.psi_prior)
        return invgamma.rvs(self.alpha_star, scale = omega_beta)

    cdef resample_psi(self, np.ndarray[np.float64_t, ndim=1] Z_t_array, np.ndarray[np.float64_t, ndim=1] B_t_array, np.ndarray[np.float64_t, ndim=1] V_t_array):
        psi_mean = utils.psi_star(self.returns, V_t_array, Z_t_array[1:-1], B_t_array[1:-1], self._mu, self.delta_t,
                                  self._kappa, self._theta, self.p_prior, self.psi_prior)
        psi_vola = np.sqrt(utils.sigma_sq_psi_star(self.returns, V_t_array, Z_t_array[1:-1], B_t_array[1:-1],
                                                   self._mu, self.delta_t, self.p_prior, self._omega))
        return np.random.normal(psi_mean, psi_vola)

    cdef resample_theta(self, np.ndarray[np.float64_t, ndim=1] Z_t_array, np.ndarray[np.float64_t, ndim=1] B_t_array, np.ndarray[np.float64_t, ndim=1] V_t_array):
        theta_mean = utils.theta_star(self.returns, V_t_array, Z_t_array[1:-1], B_t_array[1:-1], self._mu, self.delta_t,
                                      self._psi, self._kappa, self._omega, self.theta_prior,
                                      self.sigma_sq_theta_prior)
        theta_vola = np.sqrt(utils.sigma_sq_theta_star(V_t_array, self.delta_t, self._kappa,
                                                       self._omega, self.sigma_sq_theta_prior))
        return truncnorm.rvs((0 - theta_mean) / theta_vola, (5 - theta_mean) / theta_vola, loc=theta_mean,
                                    scale=theta_vola)

    cdef resample_kappa(self, np.ndarray[np.float64_t, ndim=1] Z_t_array, np.ndarray[np.float64_t, ndim=1] B_t_array, np.ndarray[np.float64_t, ndim=1] V_t_array):
        kappa_mean = utils.kappa_star(self.returns, V_t_array, Z_t_array[1:-1], B_t_array[1:-1], self._mu, self.delta_t,
                                      self._psi, self._theta, self._omega, self.kappa_prior,
                                      self.sigma_sq_kappa_prior)
        kappa_vola = np.sqrt(utils.sigma_sq_kappa_star(V_t_array, self.delta_t, self._theta,
                                                       self._omega, self.sigma_sq_kappa_prior))
        return truncnorm.rvs((0 - kappa_mean) / kappa_vola, (5 - kappa_mean) / kappa_vola, loc=kappa_mean,
                                    scale=kappa_vola)

    cdef resample_posterior_mean_jump_size(self, np.ndarray[np.float64_t, ndim=1] Z_t_array):
        mean_m_s = utils.mu_m_s_star(self.s0, self.sigma_sq_s,
                                     self.T, Z_t_array[1:-1])  # posterior mean of the mean of the jump size
        var_m_s = utils.sigma_sq_m_s_star(self.s0, self.sigma_sq_s,
                                          self.T)  # posterior variance of the mean of the jump size
        return norm.rvs(mean_m_s, np.sqrt(var_m_s))

    cdef resample_posterior_variance_jump_size(self, np.ndarray[np.float64_t, ndim=1] Z_t_array):
        beta_s_star = self.beta_s_prior + 0.5 * sum(
            (Z_t_array[1:-1] - self.mu_s) ** 2)  # parameter of the posterior variance of the jump size
        return invgamma.rvs(a = self.alpha_s_star, scale = beta_s_star)

    cdef resample_jump_intensity(self, np.ndarray[np.float64_t, ndim=1] B_t_array):
        alpha_lambda_star = self.alpha_lambda_prior + sum(B_t_array[1:-1])
        beta_lambda_star = self.beta_lambda_prior + self.T - sum(B_t_array[1:-1])
        return beta.rvs(a=alpha_lambda_star, b=beta_lambda_star)

    cpdef gibbs_sampling(self, np.ndarray[np.float64_t, ndim=1] Z_t_array, np.ndarray[np.float64_t, ndim=1] B_t_array, np.ndarray[np.float64_t, ndim=1] V_t_array):
        # ------- 1. Gibbs' sampling of model parameters -------
        # ----- start with the initialized parameters and update them using MCMC
        # (a) drift
        self._mu = self.resample_drift(Z_t_array, B_t_array, V_t_array)
        # (b) Omega
        self._omega = self.resample_omega(Z_t_array, B_t_array, V_t_array)
        # (c) psi
        self._psi = self.resample_psi(Z_t_array, B_t_array, V_t_array)
        # (d) theta
        self._theta = self.resample_theta(Z_t_array, B_t_array, V_t_array)
        # (e) kappa
        self._kappa = self.resample_kappa(Z_t_array, B_t_array, V_t_array)
        # (f) mu_s: posterior mean of the jump size
        self.mu_s = self.resample_posterior_mean_jump_size(Z_t_array)
        # (g) sigma_sq_s: posterior variance of the jump size
        self.sigma_sq_s = self.resample_posterior_variance_jump_size(Z_t_array)
        # (h) lambda: the jump intensity
        self.lambda_d = self.resample_jump_intensity(B_t_array)

    cpdef np.ndarray[np.float64_t, ndim=1] create_starting_volatility_distribution(self):
        cdef np.ndarray[np.float64_t, ndim=1] V_t = np.array(truncnorm.rvs(a=0, b=np.inf, loc=0.0225, scale=0.005, size=self.T + 2))
        return V_t

    cpdef np.ndarray[np.float64_t, ndim=1] create_starting_jump_binomial(self):
        # Per the paper, create a Bernoulli distribution with p = lambda_d
        cdef np.ndarray[np.float64_t, ndim=1] B_t = np.array(binom.rvs(n=1, p=self.lambda_d, size=self.T + 2), dtype=np.float64)
        return B_t

    cpdef np.ndarray[np.float64_t, ndim=1] create_starting_jump_sizes(self):
        cdef np.ndarray[np.float64_t, ndim=1] Z_t = np.array(np.zeros((self.T + 2,)), dtype=np.float64)
        return Z_t

    """
    
    """

    cpdef dict calibrate(self, int n_mcmc_steps, int burn_in, double rejection_rate):
        # ----- generate starting values for V using a truncated normal distribution
        #       (present-time as well as shifted backward and forward)
        cdef np.ndarray[np.float64_t, ndim=1] V_t_array = self.create_starting_volatility_distribution()
        cdef np.ndarray[np.float64_t, ndim=1] V_t_plus_1_array = np.array(np.roll(V_t_array, -1), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] V_t_minus_1_array = np.array(np.roll(V_t_array, 1), dtype=np.float64)

        # ----- generate starting values for B using the Bernoulli distribution
        #       (present-time as well as shifted backward and forward)
        cdef np.ndarray[np.float64_t, ndim=1] B_t_array = self.create_starting_jump_binomial()
        cdef np.ndarray[np.float64_t, ndim=1] B_t_plus_1_array = np.array(np.roll(B_t_array, -1), dtype=np.float64)
        B_t_plus_1_array[-1] = 0

        # ----- generate starting values for Z
        Z_t_array = self.create_starting_jump_sizes()
        cdef np.ndarray[np.float64_t, ndim=1] Z_t_plus_1_array = np.array(np.copy(Z_t_array), dtype=np.float64)

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

        for itern in tqdm(range(1, n_mcmc_steps)):
            self.gibbs_sampling(Z_t_array, B_t_array, V_t_array)
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

