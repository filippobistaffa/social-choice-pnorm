from docplex.mp.model import Model
import argparse as ap
import numpy as np
np.set_printoptions(edgeitems=1000, linewidth=1000, suppress=True, precision=5)

parser = ap.ArgumentParser()
parser.add_argument('-n', type=int, default=7, help='n')
parser.add_argument('-m', type=int, default=5, help='m')
parser.add_argument('-p', type=int, default=2, help='p')
parser.add_argument('-e', type=float, default=1e-10, help='e')
parser.add_argument('-w', type=str, default='w.csv', help='CSV file with weights')
parser.add_argument('-b', type=str, default='b.csv', help='CSV file with b vector')
parser.add_argument('-u', help='optimize only upper-triangular', action='store_true')
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

# constraints needed for pIRLS (empty)
C = np.zeros_like(A)
d = np.zeros_like(b)

#print('A =')
#print(A)
#print('b =')
#print(b.reshape(-1, 1))

if p == 2:
    cons, res, rank, a = np.linalg.lstsq(A, b, rcond=None)
    print('NUMPY L2: x =')
    if args.u:
        tmat = np.zeros((m * m,))
        tmat[idx[:l]] = cons
        print(tmat.reshape(m, m).T)
    else:
        print(cons.reshape((m, m)).T)
elif p == 1:
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
    print('CPLEX L1: x =')
    if args.u:
        tmat = np.zeros((m * m,))
        tmat[idx[:l]] = cons
        print(tmat.reshape(m, m).T)
    else:
        print(cons.reshape((m, m)).T)
elif p == -1:
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
    print('CPLEX L∞: x =')
    if args.u:
        tmat = np.zeros((m * m,))
        tmat[idx[:l]] = cons
        print(tmat.reshape(m, m).T)
    else:
        print(cons.reshape((m, m)).T)
    print('U∞ =', solution.get_value(t))
else:
    print('Initialising Julia...')
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include('pIRLS/IRLS-pNorm.jl')
    cons, it = Main.pNorm(args.e, A, b.reshape(-1, 1), p, C, d.reshape(-1, 1))
    print('JULIA L{}: x ='.format(p))
    if args.u:
        tmat = np.zeros((m * m,))
        tmat[idx[:l]] = cons
        print(tmat.reshape(m, m).T)
    else:
        print(cons.reshape((m, m)).T)

# override solution with the one from Omega
#cons = np.array([5,1,5,1.4,5,5,1,3,7,3])
#if args.u:
#    tmat = np.zeros((m * m,))
#    tmat[idx[:l]] = cons
#    print(tmat.reshape(m, m).T)
#else:
#    print(cons.reshape((m, m)))

r = np.abs(A @ cons - b)
if p != -1:
    print('U{} = {}'.format(p, np.linalg.norm(r, p)))

print()
#print('Residuals =', r)
print('Max residual =', np.max(r))
h, b = np.histogram(r, bins=np.arange(10))
print('Residuals distribution =')
print(np.vstack((h, b[:len(h)], np.roll(b, -1)[:len(h)])))
