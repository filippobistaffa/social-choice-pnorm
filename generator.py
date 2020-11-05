import numpy as np
import os

reps = 10
precision = 2
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
    for d in [b_dir, p_dir, u_dir, g_dir]:
        os.makedirs(d, exist_ok=True)
    for i in range(reps):
        dist = generate_polarised(1.2, 0.05, 2.15,0.05, 4, 3)
        p, l = np.unique(np.around(dist, precision), return_counts=True)
        np.savetxt(os.path.join(p_dir, 'p_{}.csv'.format(i)), p, fmt='%.{}f'.format(precision))
        np.savetxt(os.path.join(p_dir, 'l_{}.csv'.format(i)), l, fmt='%d')
        #
        dist = generate_unstructured(1, 2.9, 7)
        p, l = np.unique(np.around(dist, precision), return_counts=True)
        np.savetxt(os.path.join(u_dir, 'p_{}.csv'.format(i)), p, fmt='%.{}f'.format(precision))
        np.savetxt(os.path.join(u_dir, 'l_{}.csv'.format(i)), l, fmt='%d')
        #
        dist = generate_gaussian(1.4, 0.05, 7)
        p, l = np.unique(np.around(dist, precision), return_counts=True)
        np.savetxt(os.path.join(g_dir, 'p_{}.csv'.format(i)), p, fmt='%.{}f'.format(precision))
        np.savetxt(os.path.join(g_dir, 'l_{}.csv'.format(i)), l, fmt='%d')
