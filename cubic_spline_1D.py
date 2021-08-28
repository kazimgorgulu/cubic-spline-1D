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


def getData(p, x, c):
    p = np.array(p)
    x = np.array(x)
    f = np.zeros(p.shape[0])

    for k in range(p.shape[0]):
        ind = np.abs(x - p[k]).argmin()

        if (x[ind] > p[k] or ind == x.shape[0]-1) and not ind == 0:
            ind -= 1
        n = ind * 4
        f[k] = c[n] * p[k] ** 3 + c[n + 1] * p[k] ** 2 + c[n + 2] * p[k] + c[n + 3]

    return f



x = np.array([0., 1, 2, 4, 5, 6])
y = np.array([1., 2, 1, 0, -1, 0])

c = calc_coeff(x, y)
print(c)

p = np.linspace(-2,8,100)
f = np.zeros(p.size)

print('p: ', p)

for k in range(p.shape[0]):
    if p[k] >= 1:
        ind = 1
    else:
        ind = 0
    i = ind * 4
    f[k] = c[i] * p[k] ** 3 + c[i+1] * p[k] ** 2 + c[i+2] * p[k] + c[i+3]

print(f)

f2 = getData(p,x,c)

print(f2)



import matplotlib.pyplot as plt

plt.figure()
plt.plot(p,f2)

plt.plot(x,y,'o')
plt.show()













