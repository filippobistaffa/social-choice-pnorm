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
args = parser.parse_args()

p = args.p
n = args.n
m = args.m
m2 = m*m

b = np.genfromtxt(args.b)
w = np.genfromtxt(args.w)

w = np.repeat(w, m2)
b = np.multiply(b, w)

A = np.tile(np.identity(m2), (n, 1))
A = np.multiply(A, w.reshape(-1, 1))

# constraints needed for pIRLS (empty)
C = np.zeros_like(A)
d = np.zeros_like(b)

#print('A =')
#print(A)
#print('b =')
#print(b.reshape(-1, 1))

x_n, res, rank, a = np.linalg.lstsq(A, b, rcond=None)

print('NUMPY L2: x =')
print(x_n.reshape((m, m)))

print('Initialising Julia...')
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.include('pIRLS/IRLS-pNorm.jl')
x_j, it = Main.pNorm(args.e, A, b.reshape(-1, 1), p, C, d.reshape(-1, 1))
print('JULIA L{}: x ='.format(p))
print(x_j.reshape((m, m)))

r = np.abs(A @ x_j - b)
h, b = np.histogram(r, bins=np.arange(10))
print('Residuals =')
print(np.vstack((h, b[:len(h)], np.roll(b, -1)[:len(h)])))
