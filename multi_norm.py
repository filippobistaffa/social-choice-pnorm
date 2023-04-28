import argparse as ap
import cvxpy as cp
import numpy as np
np.set_printoptions(edgeitems=1000, linewidth=1000, suppress=True, precision=3)


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
    prob.solve(solver=args.S, verbose=args.v)
    return prob.value


def mLp(A, b, ps, λs, weight=True):
    wps = [λ / Lp_norm(A, b, p) if weight else λ for λ, p in zip(λs, ps)]
    # print(list(zip(wps, ps)))
    x = cp.Variable(v)
    cost = cp.sum([wp * cp.pnorm(A @ x - b, p) for wp, p in zip(wps, ps)])
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve(solver=args.S, verbose=args.v)
    res = np.abs(A @ x.value - b)
    psi = np.var([wp * np.linalg.norm(res, p) for wp, p in zip(wps, ps)])
    return x.value, res, prob.value / sum(wps), psi


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-n', type=int, default=7, help='n')
    parser.add_argument('-m', type=int, default=5, help='m')
    parser.add_argument('-w', type=str, default='w.csv', help='CSV file with weights')
    parser.add_argument('-b', type=str, default='b.csv', help='CSV file with b vector')
    parser.add_argument('-p', type=int, nargs='+', default=[1, np.inf], help='p-norms')
    parser.add_argument('-l', type=float, nargs='+', default=[], help='lambdas')
    parser.add_argument('-u', help='optimize only upper-triangular', action='store_true')
    parser.add_argument('-v', help='verbose mode', action='store_true')
    parser.add_argument('-P', help='print LaTeX code for PGFPLOTS boxplot', action='store_true')
    parser.add_argument('-M', help='perform the Mann-Whitney U test', action='store_true')
    parser.add_argument('-L', help='print LaTeX code for stats', action='store_true')
    parser.add_argument('-W', help='do not weight norms', action='store_true')
    parser.add_argument('-S', type=str, help='choose solver (either CPLEX or GUROBI)',
        choices=['CPLEX', 'GUROBI'], default='CPLEX')
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

    cons, res, u, psi = mLp(A, b, ps, λs, not(args.W))

    if args.P:
        print('\\addplot [mark=*, boxplot]')
        print('table [row sep=\\\\,y index=0] {')
        print('    data\\\\')
        import textwrap
        string = '\\\\ '.join(['{:.4f}'.format(r) for r in res]) + '\\\\'
        for line in textwrap.wrap(string, initial_indent='    ', subsequent_indent='    '):
            print(line)
        print('};')
    elif args.M:
        from scipy.stats import mannwhitneyu
        # compute the other one
        _, res1, _ = mLp(A, b, ps, λs, args.W)
        # print(np.stack((res, res1), axis=1))
        print(mannwhitneyu(res, res1))
    else:
        from scipy import stats
        nobs, (min, max), mean, variance, skewness, kurtosis = stats.describe(res)
        if args.L:
            print(f'\\num{{{min:.8f}}} & \\num{{{max:.8f}}} & \\num{{{mean:.8f}}} & \\num{{{variance:.8f}}} & \\num{{{psi:.8f}}}')
        else:
            print_consensus(cons)
            c = 13
            headers = ['U', 'min', 'max', 'avg', 'var', 'Ψ']
            print('\n+' + ('-' * c + '+') * len(headers))
            print('+' + '+'.join([s.center(c) for s in headers]) + '+')
            print('+' + ('-' * c + '+') * len(headers))
            print('+' + '+'.join(['{0:.{1}f}'.format(x, c)[:(c-2)].center(c) for x in [u, min, max, mean, variance, psi]]) + '+')
            print('+' + ('-' * c + '+') * len(headers))
