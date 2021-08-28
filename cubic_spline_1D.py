"""
1D cubic spline implementation

"""

import numpy as np
from scipy import linalg

class CubicSpline1D:
    """
    Creates a cubic spline interpolation function from given sample points
    
    Interpolation function is composed of piecewise cubic functions. 
    This function is twice-differentiable.
     
    Parameters:
    -----------
    x_data : 1D array of data points

    y_data : 1D array of values at given data points
    
    Functions:
    ---------
    __call__(xnew): Evaluates the interpolation function at given xnew points 
    
    """

    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data
        self.c = self.__calc_coeff(x, y)

    def __calc_coeff(self, x, y):
        """
        Calculates spline coefficients.
        
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

    def __call__(self, xnew):
        """ Evaluates the interpolation function at given xnew points """
        xnew = np.array(xnew)
        x = self.x
        c = self.c
        f = np.zeros(xnew.shape[0])

        for k in range(xnew.shape[0]):
            ind = np.abs(x - xnew[k]).argmin()

            if (x[ind] > xnew[k] or ind == x.shape[0] - 1) and not ind == 0:
                ind -= 1
            n = ind * 4
            f[k] = c[n] * xnew[k] ** 3 + c[n + 1] * xnew[k] ** 2 + c[n + 2] * xnew[k] + c[n + 3]

        return f



if __name__ == '__main__':
    x = np.array([0., 1, 2, 4, 5, 6])
    y = np.array([1., 2, 1, 0, -1, 0])

    
    xnew = np.linspace(-3, 8, 100)
    spline = CubicSpline1D(x,y)
    ynew = spline(xnew)

    from scipy.interpolate import CubicSpline

    
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(xnew,ynew)

    plt.plot(x,y,'o')
    plt.show()














