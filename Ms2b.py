import numpy as np
Ms = np.genfromtxt('Ms.csv', delimiter=',')
b = Ms.transpose().ravel()
np.savetxt('b.csv', b.reshape(-1, 1), delimiter=',', fmt='%f')
