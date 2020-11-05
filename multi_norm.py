import argparse as ap
import numpy as np
from general.normmin import solve
np.set_printoptions(edgeitems=1000, linewidth=1000, suppress=True, precision=4)

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
