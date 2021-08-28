"""This file presents an example for using CubicSpline1D"""

import numpy as np
import cubic_spline_1D as cs

x_data = np.array([0., 1, 2, 4, 5, 6])
y_data = np.array([1., 2, 1, 0, -1, 0])
spline = cs.CubicSpline1D(x_data, y_data)

xnew = np.linspace(-1, 7, 100)
ynew = spline(xnew)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x_data, y_data, 'o')
plt.plot(xnew, ynew)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['data points', 'interpolated points'])
plt.show()