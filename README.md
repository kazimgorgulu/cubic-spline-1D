## cubic-spline-1D

cubic-spline-1D is an implementation of spline interpolation consisting
of piecewise cubic polynomials. Resulting interpolation function
is twice continuously differentiable.
 
Natural spline boundary conditions are used in this implementation.
That means second derivatives are zero at the first and last data points.

## How to use

First create spline object using data points

``` python
import numpy as np
import cubic_spline_1D as cs

x_data = np.array([0., 1, 2, 4, 5, 6])
y_data = np.array([1., 2, 1, 0, -1, 0])
spline = cs.CubicSpline1D(x_data, y_data)
```

then calculate interpolated ynew values at query points xnew:

``` python
xnew = np.linspace(-1, 7, 100)
ynew = spline(xnew)
```

Possible improvements could be: 
1) Implementation of other boundary conditions for more flexibility. 
2) Given the coefficients are already calculated: if the input 
to spline function is monotonically
 increasing and equally spaced, a more
 efficient spline function evaluation method can be implemented.