import numpy as np
import os

seed = 0
reps = 10
precision = 1

np.random.seed(seed)
b_dir = 'distributions'
p_dir = os.path.join(b_dir, 'polarised')
u_dir = os.path.join(b_dir, 'unstructured')
g_dir = os.path.join(b_dir, 'gaussian')

def generate_polarised(mu1, sigma1, mu2, sigma2, n1, n2):
    freedom_principles = np.random.normal(mu1, sigma1, n1)
    fair_principles = np.random.normal(mu2, sigma2, n2)
    return np.concatenate([freedom_principles, fair_principles])

def generate_unstructured(low, high, number_of_principles):
    return np.random.uniform(low, high, number_of_principles)

def generate_gaussian(mu, sigma, n):
    return np.random.normal(mu, sigma, n)

def append(cumulative, directory, distribution):
    if directory in cumulative:
        cumulative[directory] = np.append(cumulative[directory], distribution)
    else:
        cumulative[directory] = distribution

if __name__ == '__main__':
    os.makedirs(b_dir, exist_ok=True)
    cumulative = {}
    for i in range(reps):
        # polarised
        for j in range(1, 7):
            dist = generate_polarised(1.05, 0.05, 2.15, 0.05, j, 7 - j)
            dist = np.around(dist, precision)
            dist = np.clip(dist, 1, 3)
            p, l = np.unique(dist, return_counts=True)
            p_dir_t = p_dir + '_{}_{}'.format(j, 7 - j)
            os.makedirs(p_dir_t, exist_ok=True)
            np.savetxt(os.path.join(p_dir_t, 'p_{}.csv'.format(i)), p, fmt='%.{}f'.format(precision))
            np.savetxt(os.path.join(p_dir_t, 'l_{}.csv'.format(i)), l, fmt='%d')
            append(cumulative, p_dir_t, dist)
        # unstructured
        dist = generate_unstructured(1, 2.9, 7)
        dist = np.around(dist, precision)
        dist = np.clip(dist, 1, 3)
        p, l = np.unique(dist, return_counts=True)
        os.makedirs(u_dir, exist_ok=True)
        np.savetxt(os.path.join(u_dir, 'p_{}.csv'.format(i)), p, fmt='%.{}f'.format(precision))
        np.savetxt(os.path.join(u_dir, 'l_{}.csv'.format(i)), l, fmt='%d')
        append(cumulative, u_dir, dist)
        # gaussian
        for mu in [1.2, 1.4, 2.15]:
            dist = generate_gaussian(mu, 0.05, 7)
            dist = np.around(dist, precision)
            dist = np.clip(dist, 1, 3)
            p, l = np.unique(dist, return_counts=True)
            g_dir_t = g_dir + '_{}'.format(mu)
            os.makedirs(g_dir_t, exist_ok=True)
            np.savetxt(os.path.join(g_dir_t, 'p_{}.csv'.format(i)), p, fmt='%.{}f'.format(precision))
            np.savetxt(os.path.join(g_dir_t, 'l_{}.csv'.format(i)), l, fmt='%d')
            append(cumulative, g_dir_t, dist)
    for directory, distribution in cumulative.items():
        p, l = np.unique(distribution, return_counts=True)
        np.savetxt(os.path.join(directory, 'cumul.csv'), np.vstack((p, l)).T,
            fmt='%.{}f'.format(precision), delimiter=',')
