import numpy as np
import matplotlib.pyplot as plt
import filterpy
from scipy.stats import binned_statistic_dd
from filterpy.kalman import KalmanFilter
from numpy.ma.extras import average
from scipy.interpolate import griddata
from matplotlib import cm

Np = 2

rand = np.random.rand(Np, Np)
prior_cov = np.dot(rand.T, rand)
al_inv_prior_cov = np.linalg.inv(0.01 * prior_cov)
inv_prior_cov = np.linalg.inv(prior_cov)

rand = np.random.rand(Np, Np)
obs_cov = np.dot(rand.T, rand)
inv_obs_cov = np.linalg.inv(obs_cov)
prior_mean = np.random.rand(Np)

num_bins = 20

#a load of equations (have fun)
def K(x_1, x_2):
    return np.exp(-(1/2) * np.dot(np.dot(np.transpose(x_1 - x_2), al_inv_prior_cov), x_1 - x_2))

def K_bold(x_1, x_2, xs):
    return K(x_1, x_2) * np.identity(xs.shape[1])

def div_K(x_1, x_2):
    return -np.dot(np.dot(al_inv_prior_cov, x_1 - x_2), K(x_1, x_2))

def H(x_1):
    return x_1

def lin_H(xs):
    return np.identity(xs.shape[1])

def grad_log_y_giv_x(x_1, y, xs):
    return np.dot(np.dot(np.transpose(lin_H(xs)), inv_obs_cov), y - H(x_1))

def grad_log_x(x_1):
    return np.dot(inv_prior_cov, x_1 - prior_mean)

def grad_log_x_hist(xs):
    maxes = np.max(xs, axis=0)
    mins = np.min(xs, axis=0)
    bins = [np.linspace(mi, ma, num_bins) for mi, ma in zip(mins, maxes)]
    counts, edges, indexes = binned_statistic_dd(
        xs,
        np.arange(len(xs)),
        bins=bins,
        expand_binnumbers=True,
        statistic="count"
    )

    grad = np.array(np.gradient(counts))
    vectorized_grad = [[grad[i][id] for i in range(grad.shape[0])] for id, value in np.ndenumerate(grad[0])]
    return vectorized_grad


def grad_log_x_giv_y(x_1, y, xs):
    return grad_log_y_giv_x(x_1, y, xs) + grad_log_x(x_1)

def ensemble_perturbation(xs, x_m):
    return np.subtract(xs, x_m)

def ensemble_mean(xs):
    return np.multiply(1 / xs.shape[0], np.sum(xs, 0))

def app_prior_cov(xs, per):
    return np.multiply(1 / (xs.shape[0] - 1), np.dot(np.transpose(per), per))

def sum_term(x, x_i, y, xs):
    return np.add(np.dot(K_bold(x_i, x, xs), grad_log_x_giv_y(x_i, y, xs)), div_K(x_i, x))

def sum_flow_eq(x, xs, y, D):
    return (1 / xs.shape[0]) * np.dot(D, np.sum([sum_term(x, xs[i], y, xs) for i in range(xs.shape[0])], 0))

def flow_matrix(xs, y, D):
    return np.array([sum_flow_eq(xs[i], xs, y, D) for i in range(xs.shape[0])])

def MA(old_av, new):
    return old_av * (19 / 20) + new / 20

xs_pff = np.random.multivariate_normal(prior_mean, prior_cov, 100)
print(grad_log_x_hist(xs_pff))

xa_pff = xs_pff

D = app_prior_cov(xs_pff, ensemble_perturbation(xs_pff, ensemble_mean(xs_pff)))

y = 5 * np.random.rand(Np)

print(f"prmean: {prior_mean}")
print(f"prcov: {prior_cov}")

print(f"pomean: {y}")

flows = np.ones(Np)

consecutive_decrease = 0
consecutive_increase = 0
last_flows = np.ones(Np)
ds = 0.05
average_diff = 100
moving_av = 0

#haha this kind of works
while moving_av < 0.9:
    flows = flow_matrix(xs_pff, y, D)
    xs_pff = xs_pff + ds * flows

    average_diff = np.average(np.abs(np.subtract(flows, last_flows)))
    if 0.0025 > average_diff > -0.0025:
        consecutive_decrease += 1
        if consecutive_decrease == 10:
            print("decrease")
            ds *= 1.4
            consecutive_decrease = 0
            consecutive_increase = 0

        moving_av = MA(moving_av, 1)
    else:
        consecutive_decrease = 0
        moving_av = MA(moving_av, 0)

    if average_diff > 0:
        consecutive_increase += 1
        if consecutive_increase == 10:
            print("increase")
            ds /= 1.4
            consecutive_decrease = 0
            consecutive_increase = 0
    else:
        consecutive_increase = 0

    last_flows = flows

    print(moving_av)

#compare against a KF
kf = KalmanFilter(2, 2)
kf.x = prior_mean
kf.P = prior_cov
kf.R = obs_cov
kf.H = np.identity(Np)
kf.F = np.identity(Np)
kf.Q = np.identity(2)

kf.update(y)
kf.predict()

correct = np.random.multivariate_normal(kf.x, kf.P, 50)

plt.axhline(y = y[0], color = 'r', linestyle = '-')
plt.axvline(x = y[1], color = 'r', linestyle = '-')
plt.scatter(xa_pff[:, 0], xa_pff[:, 1], color='g')
plt.scatter(xs_pff[:, 0], xs_pff[:, 1], color='r')
plt.axhline(y = prior_mean[0], color = 'g', linestyle = '-')
plt.axvline(x = prior_mean[1], color = 'g', linestyle = '-')
plt.scatter(correct[:, 0], correct[:, 1], color='b')


plt.tight_layout()
plt.show()