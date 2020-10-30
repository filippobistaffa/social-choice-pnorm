import argparse as ap
import numpy as np
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

def L1(A, b):
    from docplex.mp.model import Model
    model = Model("Sum of absolute residuals approximation")
    # create variables
    t = model.continuous_var_list(len(b))
    x = model.continuous_var_list(l)
    # create constraints
    I = range(len(b))   # size of b
    J = range(l)        # size of x
    for i in I:
        model.add_constraint(model.sum(A[i,j] * x[j] for j in J) - b[i] >= -t[i])
        model.add_constraint(model.sum(A[i,j] * x[j] for j in J) - b[i] <= t[i])
    model.minimize(model.sum(t))
    # optimize model
    solution = model.solve()
    cons = np.zeros((l,))
    for j in J:
        cons[j] = solution.get_value(x[j])
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r, 1)

def L2(A, b):
    cons, res, rank, a = np.linalg.lstsq(A, b, rcond=None)
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r)

def Linf(A, b):
    from docplex.mp.model import Model
    model = Model("Chebyshev approximation")
    # create variables
    t = model.continuous_var()
    x = model.continuous_var_list(l)
    # create constraints
    I = range(len(b))   # size of b
    J = range(l)        # size of x
    for i in I:
        model.add_constraint(model.sum(A[i,j] * x[j] for j in J) - b[i] >= -t)
        model.add_constraint(model.sum(A[i,j] * x[j] for j in J) - b[i] <= t)
    model.minimize(t)
    # optimize model
    solution = model.solve()
    cons = np.zeros((l,))
    for j in J:
        cons[j] = solution.get_value(x[j])
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r, np.inf)

def IRLS(A, b, p, max_iter=int(1e6), e=1e-3, d=1e-4):
    n = A.shape[0]
    D = np.repeat(d, n)
    W = np.diag(np.repeat(1, n))
    x = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ b # initial LS solution
    for i in range(max_iter):
        W_ = np.diag(np.power(np.maximum(np.abs(b - A @ x), D), p - 2))
        x_ = np.linalg.inv(A.T @ W_ @ A) @ A.T @ W_ @ b # reweighted LS solution
        e_ = sum(abs(x - x_))
        #print(e_)
        if e_ < e:
            break
        else:
            W = W_
            x = x_
    r = np.abs(A @ x - b)
    return x, r, np.linalg.norm(r, p)

def Lp(A, b, p):
    if p >= 2: # pIRLS implementation (NIPS 2019)
        # uncomment to compare with vanilla implementation
        #if p < 3: # vanilla does not converge for p >= 3
        #    cons, _, _ = IRLS(A, b, p)
        #    print_consensus(cons)
        from julia.api import LibJulia
        api = LibJulia.load()
        api.sysimage = os.path.dirname(os.path.realpath(__file__)) + '/sys.so'
        api.init_julia()
        from julia import Main
        Main.include('pIRLS/IRLS-pNorm.jl')
        # constraints needed for pIRLS (empty)
        C = np.zeros_like(A)
        d = np.zeros_like(b)
        epsilon = 1e-10
        cons, it = Main.pNorm(epsilon, A, b.reshape(-1, 1), p, C, d.reshape(-1, 1))
        r = np.abs(A @ cons - b)
        return cons, r, np.linalg.norm(r, p)
    else: # vanilla IRLS implementation
        return IRLS(A, b, p)

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-n', type=int, default=7, help='n')
    parser.add_argument('-m', type=int, default=5, help='m')
    parser.add_argument('-p', type=float, default=2, help='p')
    parser.add_argument('-e', type=float, default=5e-2, help='e')
    parser.add_argument('-w', type=str, default='w.csv', help='CSV file with weights')
    parser.add_argument('-b', type=str, default='b.csv', help='CSV file with b vector')
    parser.add_argument('-i', type=str, help='computes equivalent p given an input consensus')
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
        _, _, ua = L1(A, b)
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
        cons_1, r_1, u_1 = L1(A, b)
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
                print('p = {:.2f} not improving anymore, stopping!'.format(i))
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
                print('p = {:.2f} not improving anymore, stopping!'.format(i))
                break
            else:
                print('p = {:.2f}'.format(i))
                print('Distance = {:.4f}'.format(dist))
                print('Current best distance = {:.4f}'.format(best))
                best = dist
    else:
        if p == 2:
            cons, r, u = L2(A, b)
            print_consensus(cons)
        elif p == 1:
            cons, r, u = L1(A, b)
            print_consensus(cons)
        elif p == -1:
            cons, r, u = Linf(A, b)
            print_consensus(cons)
        else:
            cons, r, u = Lp(A, b, p)
            print_consensus(cons)

        # override solution with the one from Omega
        #cons = np.array([5,1,5,1.4,5,5,1,3,7,3])
        #print_consensus(cons)

        if p != -1:
            print('U{} = {:.4f}'.format(p, u))
        else:
            print('U∞ = {:.4f}'.format(u))

        print()
        #print('Residuals =', r)
        print('Max residual = {:.4f}'.format(np.max(r)))
        h, b = np.histogram(r, bins=np.arange(10))
        print('Residuals distribution =')
        print(np.vstack((h, b[:len(h)], np.roll(b, -1)[:len(h)])))
