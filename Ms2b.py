import numpy as np
np.set_printoptions(edgeitems=1000, linewidth=1000, suppress=True)
Ms = np.genfromtxt('Ms.csv', delimiter=',')
b = Ms.transpose().ravel()
np.savetxt('b.csv', b.reshape(-1, 1), delimiter=',', fmt='%f')
