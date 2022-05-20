import numpy as np


n = 7
m = 5
seed = 0
np.random.seed(seed)


if __name__ == '__main__':
    b = []
    for i in range(n):
        rank = np.arange(5)
        np.random.shuffle(rank)
        #print(rank)
        r = np.zeros((m, m), dtype=int)
        for j, a in enumerate(rank):
            for k, b in enumerate(rank[j + 1:], start=j + 1):
                r[a, b] = 1
        for a in r.ravel():
            print(a)
