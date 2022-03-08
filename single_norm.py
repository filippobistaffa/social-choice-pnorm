import argparse as ap
import numpy as np
import cvxpy as cp
import os
np.set_printoptions(edgeitems=1000, linewidth=1000, suppress=True, precision=4)

def print_consensus(cons):
    print('Rs =')
    if args.u:
        tmat = np.zeros((m * m,))
        tmat[idx[:l]] = cons
        print(tmat.reshape(m, m).T)
    else:
        print(cons.reshape((m, m)).T)

def Lp(A, b, p):
    x = cp.Variable(l)
    cost = cp.pnorm(A @ x - b, p)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve(solver='CPLEX', verbose=False, cplex_params={})
    res = np.abs(A @ x.value - b)
    return x.value, res, prob.value

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-n', type=int, default=7, help='n')
    parser.add_argument('-m', type=int, default=5, help='m')
    parser.add_argument('-p', type=float, default=2, help='p')
    parser.add_argument('-e', type=float, default=5e-2, help='e')
    parser.add_argument('-w', type=str, default='w.csv', help='CSV file with weights')
    parser.add_argument('-b', type=str, default='b.csv', help='CSV file with b vector')
    parser.add_argument('-i', type=str, help='computes equivalent p given an input consensus')
    parser.add_argument('-o', type=str, help='write consensus to file')
    parser.add_argument('-u', help='optimize only upper-triangular', action='store_true')
    parser.add_argument('-l', help='compute the limit p', action='store_true')
    parser.add_argument('-t', help='compute the threshold p', action='store_true')
    args = parser.parse_args()

    p = args.p
    n = args.n
    m = args.m

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

    #print('A =')
    #print(A)
    #print('b =')
    #print(b.reshape(-1, 1))

    if args.l:
        _, _, ua = Lp(A, b, 1)
        print('U1 = {:.4f}'.format(ua))
        p = 1
        incr = 0.01
        while True:
            p += incr
            _, _, ub = Lp(A, b, p)
            print('U{:.2f} = {:.4f}'.format(p, ub), end='')
            du = abs(ua - ub)
            slope = du / incr
            if du < args.e:
                print(' (ΔU = {:.4f} < {})'.format(du, args.e))
                break
            else:
                print(' (ΔU = {:.4f} > {})'.format(du, args.e))
                ua = ub
    elif args.t:
        cons_1, r_1, u_1 = Lp(A, b, 1)
        cons_l, r_l, u_l = Lp(A, b, p)
        #print('L1:')
        #print_consensus(cons_1)
        #print('L{:.2f}:'.format(p))
        #print_consensus(cons_l)
        diff = np.inf
        incr = 0.01
        for i in np.arange(1 + incr, p, incr):
            cons, r, u = Lp(A, b, i)
            #print_consensus(cons)
            dist_1p = np.linalg.norm(cons_1 - cons, i)
            dist_pl = np.linalg.norm(cons_l - cons, i)
            if (abs(dist_1p - dist_pl) > diff):
                print('Not improving anymore, stopping!'.format(i))
                break
            else:
                print('p = {:.2f}'.format(i))
                print('Distance L1<-->L{:.2f} = {:.4f}'.format(i, dist_1p))
                print('Distance L{:.2f}<-->L{:.2f} = {:.4f}'.format(i, p, dist_pl))
                print('Difference (L1<-->L{:.2f}) - (L{:.2f}<-->L{:.2f}) = {:.4f}'.format(i, i, p, abs(dist_1p - dist_pl)))
                print('Current best difference (L1<-->L{:.2f}) - (L{:.2f}<-->L{:.2f}) = {:.4f}'.format(i, i, p, diff))
                diff = abs(dist_1p - dist_pl)
    elif args.i:
        cons = np.genfromtxt(args.i)
        print_consensus(cons)
        best = np.inf
        incr = 0.01
        for i in np.arange(1 + incr, p, incr):
            x, r, u = Lp(A, b, i)
            #print_consensus(x)
            dist = np.linalg.norm(cons - x, i)
            if (dist > best):
                print('Not improving anymore, stopping!'.format(i))
                break
            else:
                print('p = {:.2f}'.format(i))
                print('Distance = {:.4f}'.format(dist))
                print('Current best distance = {:.4f}'.format(best))
                best = dist
    else:
        cons, r, u = Lp(A, b, np.inf if p < 0 else p)
        print_consensus(cons)
        # override solution with the one from Omega
        #cons = np.array([5,1,5,1.4,5,5,1,3,7,3])
        #print_consensus(cons)
        print('U{} = {:.4f}\n'.format('∞' if p < 0 else p, u))
        #print('Residuals =', r)
        print('Max residual = {:.4f}'.format(np.max(r)))
        h, b = np.histogram(r, bins=np.arange(10))
        print('Residuals distribution =')
        print(np.vstack((h, b[:len(h)], np.roll(b, -1)[:len(h)])))
        if args.o:
            np.savetxt(args.o, cons, fmt='%.20f')
