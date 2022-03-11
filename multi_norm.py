import argparse as ap
import cvxpy as cp
import numpy as np
np.set_printoptions(edgeitems=1000, linewidth=1000, suppress=True, precision=4)


def print_consensus(cons):
    print('Rs =')
    if args.u:
        tmat = np.zeros((m * m,))
        tmat[idx[:v]] = cons
        print(tmat.reshape(m, m).T)
    else:
        print(cons.reshape((m, m)).T)


def Lp_norm(A, b, p):
    x = cp.Variable(v)
    cost = cp.pnorm(A @ x - b, p)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve(solver='CPLEX', verbose=False)
    return prob.value


def mLp(A, b, ps, λs, weight=True):
    wps = [λ / Lp_norm(A, b, p) if weight else λ for λ, p in zip(λs, ps)]
    # print(list(zip(wps, ps)))
    x = cp.Variable(v)
    cost = cp.sum([wp * cp.pnorm(A @ x - b, p) for wp, p in zip(wps, ps)])
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve(solver='CPLEX', verbose=False, cplex_params={})
    res = np.abs(A @ x.value - b)
    # print([wp * np.linalg.norm(res, p) for wp, p in zip(wps, ps)])
    return x.value, res, prob.value / sum(wps)


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-n', type=int, default=7, help='n')
    parser.add_argument('-m', type=int, default=5, help='m')
    parser.add_argument('-w', type=str, default='w.csv', help='CSV file with weights')
    parser.add_argument('-b', type=str, default='b.csv', help='CSV file with b vector')
    parser.add_argument('-p', type=int, nargs='+', default=[1, np.inf], help='p-norms')
    parser.add_argument('-l', type=float, nargs='+', default=[], help='lambdas')
    parser.add_argument('-o', type=str, help='write consensus to file')
    parser.add_argument('-u', help='optimize only upper-triangular', action='store_true')
    parser.add_argument('-v', help='verbose mode', action='store_true')
    parser.add_argument('--histogram', help='show histogram of residuals', action='store_true')
    parser.add_argument('--boxplot', help='show boxplot of residuals', action='store_true')
    parser.add_argument('--no-weights', help='do not weight norms', action='store_true')
    args = parser.parse_args()

    n = args.n
    m = args.m
    ps = np.atleast_1d(args.p)
    ps = np.where(ps == -1, np.inf, ps)
    λs = np.ones_like(ps)
    nλs = min(len(λs), len(args.l))
    λs[:nλs] = args.l[:nλs]

    if args.u:
        idx = []
        for i in range(n):
            for j in range(m):
                for k in range(j):
                    idx.append(k + m * j + m * m * i)
        v = int(m * (m - 1) / 2)
        b = np.genfromtxt(args.b)[idx]
    else:
        b = np.genfromtxt(args.b)
        v = m * m

    w = np.genfromtxt(args.w)
    w = np.repeat(w, v)
    b = np.multiply(b, w)
    A = np.tile(np.identity(v), (n, 1))
    A = np.multiply(A, w.reshape(-1, 1))

    if args.v:
        print('A =')
        print(A)
        print('b =')
        print(b.reshape(-1, 1))
        print('p =', ps)
        print('λ =', λs)

    if args.boxplot:
        _, r, _ = mLp(A, b, ps, λs, False)
        _, rw, _ = mLp(A, b, ps, λs, True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title('Boxplots of residuals for p = {}'.format(ps))
        ax.boxplot([r, rw])
        ax.set_xticklabels(['Not Weighted', 'Weighted'])
        plt.show()
    else:
        cons, r, u = mLp(A, b, ps, λs, not(args.no_weights))
        print_consensus(cons)

        if args.histogram:
            print('Residuals distribution:')
            try:
                import plotille
                print(plotille.hist(r))
            except ImportError:
                h, b = np.histogram(r, bins=np.arange(10))
                print(np.vstack((h, b[:len(h)], np.roll(b, -1)[:len(h)])))

        if args.o:
            np.savetxt(args.o, cons, fmt='%.20f')
