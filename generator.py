import numpy as np

def generate_polarised(mu1,sigma1,mu2,sigma2,n1,n2):
    freedom_principles = np.random.normal(mu1, sigma1, n1)
    fair_principles = np.random.normal(mu2, sigma2, n2)
    return np.concatenate([freedom_principles, fair_principles])

def generate_unstructured(low,high,number_of_principles):
    return np.random.uniform(low,high,number_of_principles)

def generate_consensus(mu,sigma,n):
    return np.random.normal(mu, sigma, n)

if __name__ == '__main__':
    polarised_distribution = generate_polarised(1.2,0.05,2.15,0.05,4,3)
    random_distribution = generate_unstructured(1,2.9,7)
    consensus_distribution = generate_consensus(1.4,0.05,7)
