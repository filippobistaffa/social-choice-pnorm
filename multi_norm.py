import warnings
import argparse as ap
import numpy as np
np.set_printoptions(edgeitems=1000, linewidth=1000, suppress=True, precision=4)

def residue_norm(e_list=None):
    residue = 0.0
    for e in e_list:
        residue = residue + np.linalg.norm(e)
    return residue

def solve(A_list=None, b_list=None, lambda_list=None, p_list=None, max_iter=1000000000, tol=1.0e-8):
    """
    Solves a general norm minimization problem
        minimize_x \sum_k \lambda_k * ||A_k x - b_k||_{p_k}^{p_k}

    :param A_list: List of matrices A_k \in \mathbb{R}^{m_k \times n}
    :param b_list: List of vectors b_k \in \mathbb{R}^{m_k}
    :param lambda_list: List of weighting scalar \lambda_k \in \mathbb{R}
    :param p_list: List of norm indicators p_k \ in \mathbb{R}
    :param max_iter: Maximum number of iterations
    :param tol: Tolerance
    :return: x \in \mathbb{R}^n
    :return: residue norm
    :return: number of iterations spent for computation
    """

    alpha = 0    #1.0e-8   # small value for regularizing the weighted least squares
    eps = 1.0e-8    # small value for avoiding zero-division in weight update

    if A_list is None or b_list is None or lambda_list is None or p_list is None:
        raise ValueError("Required argument is empty")
    K = len(A_list)
    if K != len(b_list) or K != len(lambda_list) or K != len(p_list):
        raise ValueError("Inconsistent arguments")
    if any(lambda_list[a] < 0 for a in range(len(lambda_list))):
        raise ValueError("lambda needs to be all positive")
    if any(p_list[a] < 0 for a in range(len(p_list))):
        raise ValueError("p needs to be all positive")
    for index in range(len(b_list)):
        if b_list[index].ndim == 1:
            b_list[index] = b_list[index][:, np.newaxis]

    n = A_list[0].shape[1]    # domain dim
    x_old = np.ones((n, 1))
    In = np.identity(n)
    w_list = []
    e_list = []
    for k in range(K):
        w_list.append(np.identity(A_list[k].shape[0]))
        e_list.append(np.zeros((A_list[k].shape[0], 1)))

    ite = 0
    while ite < max_iter:
        ite = ite + 1
        C = alpha * In    # n \times n matrix
        d = np.zeros((n, 1))    # n-vector
        # Create a normal equation of the problem
        for k in range(K):
            C = C + p_list[k] * lambda_list[k] * np.dot(np.dot(A_list[k].T, w_list[k]), A_list[k])
            d = d + p_list[k] * lambda_list[k] * np.dot(np.dot(A_list[k].T, w_list[k]), b_list[k])
        x = np.linalg.solve(C, d)
        for k in range(K):
            e_list[k] = b_list[k] - A_list[k].dot(x)
        # stopping criteria
        if np.linalg.norm(x - x_old) < tol:
            return x.ravel(), residue_norm(e_list), ite
        else:
            #print(x[0], x[1])
            x_old = x
        # update weights
        for k in range(K):
            w_list[k] = np.diag(
               np.asarray(1.0 / np.maximum(np.power(np.fabs(e_list[k]), 2.0 - p_list[k]), eps))[:, 0])

    warnings.warn("Exceeded the maximum number of iterations")
    return x.ravel(), residue_norm(e_list), ite

def print_consensus(cons):
    print('Rs =')
    if args.u:
        tmat = np.zeros((m * m,))
        tmat[idx[:l]] = cons
        print(tmat.reshape(m, m).T)
    else:
        print(cons.reshape((m, m)).T)

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-n', type=int, default=7, help='n')
    parser.add_argument('-m', type=int, default=5, help='m')
    parser.add_argument('-w', type=str, default='w.csv', help='CSV file with weights')
    parser.add_argument('-b', type=str, default='b.csv', help='CSV file with b vector')
    parser.add_argument('-p', type=str, default='p.csv', help='CSV file with norms')
    parser.add_argument('-l', type=str, default='l.csv', help='CSV file with lambdas')
    parser.add_argument('-o', type=str, help='write consensus to file')
    parser.add_argument('-u', help='optimize only upper-triangular', action='store_true')
    parser.add_argument('-v', help='verbose mode', action='store_true')
    args = parser.parse_args()

    n = args.n
    m = args.m

    w = np.genfromtxt(args.w)
    p = np.atleast_1d(np.genfromtxt(args.p))

    if len(p) < 2:
        raise ValueError('Specify at least 2 norm values')

    if args.u:
        idx = []
        for i in range(n):
            for j in range(m):
                for k in range(j):
                    idx.append(k + m * j + m * m * i)
        l = int(m * (m - 1) / 2)
        b = np.genfromtxt(args.b)[idx]
    else:
        b = np.genfromtxt(args.b)
        l = m * m

    w = np.repeat(w, l)
    b = np.multiply(b, w)

    A = np.tile(np.identity(l), (n, 1))
    A = np.multiply(A, w.reshape(-1, 1))

    if args.v:
        print('A =')
        print(A)
        print('b =')
        print(b.reshape(-1, 1))
        print('p =', p)

    cons, u, it = solve(A_list=[A] * len(p), b_list=[b] * len(p), p_list=p,
                        lambda_list=np.atleast_1d(np.genfromtxt(args.l))[:len(p)])
    print_consensus(cons)
    print('U{} = {:.3f}'.format(p, u))
    print()
    r = np.abs(A @ cons - b)
    #print('Residuals =', r)
    print('Max residual = {:.3f}'.format(np.max(r)))
    h, b = np.histogram(r, bins=np.arange(10))
    print('Residuals distribution =')
    print(np.vstack((h, b[:len(h)], np.roll(b, -1)[:len(h)])))

    if args.o:
        np.savetxt(args.o, cons, fmt='%.20f')
