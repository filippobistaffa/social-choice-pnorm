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

if __name__ == '__main__':
    os.makedirs(b_dir, exist_ok=True)
    for i in range(reps):
        # polarised
        for j in range(1, 7):
            dist = generate_polarised(1.05, 0.05, 2.15, 0.05, j, 7 - j)
            p, l = np.unique(np.around(dist, precision), return_counts=True)
            p_dir_t = p_dir + '_{}_{}'.format(j, 7 - j)
            os.makedirs(p_dir_t, exist_ok=True)
            np.savetxt(os.path.join(p_dir_t, 'p_{}.csv'.format(i)), p, fmt='%.{}f'.format(precision))
            np.savetxt(os.path.join(p_dir_t, 'l_{}.csv'.format(i)), l, fmt='%d')
        # unstructured
        dist = generate_unstructured(1, 2.9, 7)
        p, l = np.unique(np.around(dist, precision), return_counts=True)
        os.makedirs(u_dir, exist_ok=True)
        np.savetxt(os.path.join(u_dir, 'p_{}.csv'.format(i)), p, fmt='%.{}f'.format(precision))
        np.savetxt(os.path.join(u_dir, 'l_{}.csv'.format(i)), l, fmt='%d')
        # gaussian
        dist = generate_gaussian(1.4, 0.05, 7)
        p, l = np.unique(np.around(dist, precision), return_counts=True)
        os.makedirs(g_dir, exist_ok=True)
        np.savetxt(os.path.join(g_dir, 'p_{}.csv'.format(i)), p, fmt='%.{}f'.format(precision))
        np.savetxt(os.path.join(g_dir, 'l_{}.csv'.format(i)), l, fmt='%d')
