"""
1D cubic spline implementation

"""

import numpy as np
from scipy import linalg


def calc_coeff(x, y):
    """
    Calculates spline coefficients from given sample data.

    Parameters:
    -----------
    x : 1D array of data points

    y : 1D array of values at given data points
    """
    n = x.shape[0]
    m = 4 * (n - 1)

    A = np.zeros((m, m))
    b = np.zeros((m, 1))

    cursor = 0
    for i in range(n):
        if i == 0:
            A[cursor, i * 4: (i + 1) * 4] = [x[0] ** 3, x[0] ** 2, x[0], 1]
            b[cursor] = y[i]
            cursor += 1
            A[cursor, i * 4: (i + 1) * 4] = [6 * x[0], 2, 0, 0]
            b[cursor] = 0
            cursor += 1
        elif i == n - 1:
            A[cursor, (i - 1) * 4: i * 4] = [x[-1] ** 3, x[-1] ** 2, x[-1], 1]
            b[cursor] = y[i]
            cursor += 1
            A[cursor, (i - 1) * 4: i * 4] = [6 * x[-1], 2, 0, 0]
            b[cursor] = 0
            cursor += 1
        else:
            A[cursor, (i - 1) * 4: i * 4] = [x[i] ** 3, x[i] ** 2, x[i], 1]
            b[cursor] = y[i]
            cursor += 1
            A[cursor, i * 4: (i + 1) * 4] = [x[i] ** 3, x[i] ** 2, x[i], 1]
            b[cursor] = y[i]
            cursor += 1
            A[cursor, (i - 1) * 4: (i + 1) * 4] = [3 * x[i] ** 2, 2 * x[i], 1, 0, -3 * x[i] ** 2, -2 * x[i], -1, 0, ]
            b[cursor] = 0
            cursor += 1
            A[cursor, (i - 1) * 4: (i + 1) * 4] = [6 * x[i], 2, 0, 0, -6 * x[i], -2, 0, 0, ]
            b[cursor] = 0
            cursor += 1
    return linalg.solve(A,b)


def getData(p, min_x, max_x, len_x,c):
    p = np.array(p)
    d = (max_x-min_x)/(len_x-1)
    print(p)
    # print(np.array((p-min_x)/d).astype(int))
    ind = np.array((p-min_x)/d).astype(int)
    print(ind)
    f = np.zeros(ind.shape[0])
    print(f)
    for k in range(ind.shape[0]):
        i = ind * 4
        print(c[i] * p[k] ** 3 + c[i + 1] * p[k] ** 2 + c[i + 2] * p[k] + c[i + 3])
        print(c[i])
        f[k] = c[i] * p[k] ** 3 + c[i + 1] * p[k] ** 2 + c[i + 2] * p[k] + c[i + 3]

    return f



x = np.array([0., 1, 2])
y = np.array([1., 1.5, 1])

c = calc_coeff(x, y)
print(c)
# ind = getData([0,1],min(x), max(x), len(x),c)

# print(ind)
p = np.linspace(0,4,100)
f = np.zeros(p.size)

ind = 0


for k in range(p.shape[0]):
    if p[k] >= 1:
        ind = 1
    else:
        ind = 0
    i = ind * 4
    f[k] = c[i] * p[k] ** 3 + c[i+1] * p[k] ** 2 + c[i+2] * p[k] + c[i+3]


import matplotlib.pyplot as plt

plt.figure()
plt.plot(p,f)
plt.plot(x,y,'o')
plt.show()













