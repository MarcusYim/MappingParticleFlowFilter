import numpy as np

prior_cov = np.array([[0.2, 1], [0.7, 0.1]])
inv_prior_cov = np.linalg.inv(prior_cov)
al_inv_prior_cov = 0.01 * inv_prior_cov
obs_cov = np.array([[0.2, 1], [1, 0.3]])
inv_obs_cov = np.linalg.inv(obs_cov)
prior_mean = np.array([1, 1])

def K(x_1, x_2):
    return np.exp(-(1/2) * np.dot(np.dot(np.transpose(x_1 - x_2), al_inv_prior_cov), x_1 - x_2))

def K_bold(x_1, x_2, xs):
    return K(x_1, x_2) * np.identity(xs.shape[1])

def div_K(x_1, x_2):
    return -np.dot(np.dot(np.transpose(al_inv_prior_cov), x_1 - x_2), K(x_1, x_2))

def H(x_1):
    return x_1

def lin_H(x_1):
    return np.identity(2)

def grad_log_y_giv_x(x_1, y):
    return np.dot(np.dot(np.transpose(lin_H(x_1)), inv_obs_cov), y - H(x_1))

def grad_log_x(x_1):
    return np.dot(inv_prior_cov, x_1 - prior_mean)

def grad_log_x_giv_y(x_1, y):
    return grad_log_y_giv_x(x_1, y) + grad_log_x(x_1)

def ensemble_perturbation(xs, x_m):
    return np.subtract(xs, x_m)

def ensemble_mean(xs):
    return np.multiply(1 / xs.shape[0], np.sum(xs, 0))

def prior_cov(xs, per):
    return np.multiply(1 / (xs.shape[0] - 1), np.dot(np.transpose(per), per))

def sum_term(x, x_i, y, xs):
    return np.add(np.dot(K_bold(x_i, x, xs), grad_log_x_giv_y(x_i, y)), div_K(x_i, x))

def sum_flow_eq(x, xs, y, D):
    return (1 / xs.shape[0]) * np.dot(D, np.sum([sum_term(x, xs[i], y, xs) for i in range(xs.shape[0])], 0))

xs = np.array([[1, 4], [2, 1], [5,2]])

D = prior_cov(xs, ensemble_perturbation(xs, ensemble_mean(xs)))

print(sum_flow_eq(xs[2], xs, np.array([1, 1]), D))

