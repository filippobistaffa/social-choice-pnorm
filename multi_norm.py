import argparse as ap
import cvxpy as cp
import numpy as np
np.set_printoptions(edgeitems=1000, linewidth=1000, suppress=True, precision=4)

def print_consensus(cons):
    print('Rs =')
    if args.u:
        tmat = np.zeros((m * m,))
        tmat[idx[:l]] = cons
        print(tmat.reshape(m, m).T)
    else:
        print(cons.reshape((m, m)).T)

def Lp_norm(A, b, p):
    x = cp.Variable(l)
    cost = cp.pnorm(A @ x - b, p)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve(solver='CPLEX', verbose=False)
    return prob.value

def mLp(A, b, ps, λs):
    wps = [λ / Lp_norm(A, b, p) for λ, p in zip(λs, ps)]
    #wps = [λ for λ, p in zip(λs, ps)]
    #print(list(zip(wps, ps)))
    x = cp.Variable(l)
    cost = cp.sum([wp * cp.pnorm(A @ x - b, p) for wp, p in zip(wps, ps)])
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve(solver='CPLEX', verbose=False, cplex_params={})
    res = np.abs(A @ x.value - b)
    #print([wp * np.linalg.norm(res, p) for wp, p in zip(wps, ps)])
    return x.value, res, prob.value / sum(wps)

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
    ps = np.atleast_1d(np.genfromtxt(args.p))
    ps = np.where(ps == -1, np.inf, ps)
    λs = np.atleast_1d(np.genfromtxt(args.l))

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

    w = np.genfromtxt(args.w)
    w = np.repeat(w, l)
    b = np.multiply(b, w)

    A = np.tile(np.identity(l), (n, 1))
    A = np.multiply(A, w.reshape(-1, 1))

    if args.v:
        print('A =')
        print(A)
        print('b =')
        print(b.reshape(-1, 1))
        print('p =', ps)
        print('λ =', λs)

    cons, r, u = mLp(A, b, ps, λs)
    print_consensus(cons)
    print('U{} = {:.4f}\n'.format(ps, u))
    #print('Residuals =', r)
    print('Max residual = {:.4f}'.format(np.max(r)))
    h, b = np.histogram(r, bins=np.arange(10))
    print('Residuals distribution =')
    print(np.vstack((h, b[:len(h)], np.roll(b, -1)[:len(h)])))

    if args.o:
        np.savetxt(args.o, cons, fmt='%.20f')
