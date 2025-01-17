# Partial Derivatives and Multiple Integrations


## 1. Limit and Continuity

### Definition
For a multivariable function \( f(x, y) \), the **limit** as \( (x, y) \to (a, b) \) is:

\[
\lim_{(x, y) \to (a, b)} f(x, y) = L
\]

if \( f(x, y) \) approaches \( L \) from all directions.

A function is **continuous** at \( (a, b) \) if:
1. \( f(a, b) \) is defined.
2. \( \lim_{(x, y) \to (a, b)} f(x, y) \) exists.
3. \( \lim_{(x, y) \to (a, b)} f(x, y) = f(a, b) \).

### Real-World Example
Temperature at a location \( f(x, y) \) is continuous if small changes in \( x \) and \( y \) result in small changes in \( f(x, y) \).

### Example (Python Code):
```python
import numpy as np
import matplotlib.pyplot as plt

# Example of a continuous function f(x, y) = sin(x) * cos(y)
def f(x, y):
    return np.sin(x) * np.cos(y)

x = np.linspace(-np.pi, np.pi, 100)
y = np.linspace(-np.pi, np.pi, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.title('Continuous Function: f(x, y) = sin(x) * cos(y)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

---

## 2. Partial Derivatives

### Definition
The partial derivative of \( f(x, y) \) with respect to \( x \) is:

\[
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x + h, y) - f(x, y)}{h}
\]

Similarly, with respect to \( y \):

\[
\frac{\partial f}{\partial y} = \lim_{h \to 0} \frac{f(x, y + h) - f(x, y)}{h}
\]

### Real-World Example
In a heatmap of temperature \( f(x, y) \):

- \( \frac{\partial f}{\partial x} \): Change in temperature in the east-west direction.
- \( \frac{\partial f}{\partial y} \): Change in temperature in the north-south direction.

### Example (Python Code):
```python
# Compute partial derivatives numerically
def f(x, y):
    return x**2 + y**2

def partial_x(x, y):
    return 2 * x

def partial_y(x, y):
    return 2 * y

x, y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
z = f(x, y)
dx = partial_x(x, y)
dy = partial_y(x, y)

plt.quiver(x, y, dx, dy, color='red')
plt.contour(x, y, z, cmap='viridis')
plt.title('Gradient Field of f(x, y) = x^2 + y^2')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

---

## 3. Tangent Planes

### Definition
The tangent plane to \( z = f(x, y) \) at \( (a, b) \) is:

\[
z - f(a, b) = \frac{\partial f}{\partial x}(a, b)(x - a) + \frac{\partial f}{\partial y}(a, b)(y - b)
\]

### Real-World Example
In terrain modeling, the tangent plane gives the slope and direction of a hill at a specific point.

### Example (Python Code):
```python
# Tangent plane to f(x, y) = x^2 + y^2 at (1, 1)
def tangent_plane(x, y):
    return 2 * (x - 1) + 2 * (y - 1) + 2

x, y = np.meshgrid(np.linspace(0, 2, 20), np.linspace(0, 2, 20))
z = f(x, y)
tz = tangent_plane(x, y)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, alpha=0.6, cmap='viridis', label='Surface')
ax.plot_surface(x, y, tz, alpha=0.4, color='red', label='Tangent Plane')
ax.set_title('Tangent Plane to f(x, y) at (1, 1)')
plt.show()
```

---

## 4. Maximum and Minimum Values

### Definition
Critical points occur where \( \frac{\partial f}{\partial x} = 0 \) and \( \frac{\partial f}{\partial y} = 0 \). 

- **Maximum**: \( f(x, y) \) reaches a peak.
- **Minimum**: \( f(x, y) \) reaches a trough.
- **Saddle Point**: \( f(x, y) \) changes direction.

### Example (Python Code):
```python
from scipy.optimize import minimize

# Find critical points of f(x, y) = x^2 + y^2
def f_to_minimize(x):
    return x[0]**2 + x[1]**2

result = minimize(f_to_minimize, [1, 1])
print("Critical Point:", result.x)
```

---

## 5. Multiple Integrals

## Overview
Multiple integration extends the concept of single-variable integration to functions of two or more variables. It is used to compute the volume under a surface, areas, or to solve higher-dimensional problems in physics, engineering, and mathematics.

### Types of Multiple Integrals:
1. **Double Integral**: Integration over a two-dimensional region.
2. **Triple Integral**: Integration over a three-dimensional region.

---

## Double Integrals
### Definition
A double integral is used to compute the volume under a surface \( z = f(x, y) \) over a two-dimensional region \( R \).

The general form is:

\[
\iint_R f(x, y) \ dA
\]

where \( dA \) is the area element (typically \( dx \, dy \) or \( dy \, dx \)).

### Example
Evaluate the double integral:

\[
\iint_R (x + y) \ dA
\]

where \( R \) is the rectangle defined by \( 0 \leq x \leq 2 \) and \( 0 \leq y \leq 3 \).

#### Solution:

\[
\iint_R (x + y) \ dA = \int_0^2 \int_0^3 (x + y) \ dy \ dx
\]

1. Integrate with respect to \( y \):

\[
\int_0^3 (x + y) \ dy = \left[ xy + \frac{y^2}{2} \right]_0^3 = 3x + \frac{9}{2}
\]

2. Integrate with respect to \( x \):

\[
\int_0^2 \left( 3x + \frac{9}{2} \right) \ dx = \left[ \frac{3x^2}{2} + \frac{9x}{2} \right]_0^2 = \frac{12}{2} + \frac{18}{2} = 15
\]

Thus, the value of the integral is \( 15 \).

---

## Triple Integrals
### Definition
A triple integral is used to compute the volume of a region in three-dimensional space or to integrate functions over a three-dimensional region.

The general form is:

\[
\iiint_V f(x, y, z) \ dV
\]

where \( dV \) is the volume element (typically \( dx \, dy \, dz \), or any permutation).

### Example
Evaluate the triple integral:

\[
\iiint_V xyz \ dV
\]

where \( V \) is the cuboid defined by \( 0 \leq x \leq 1 \), \( 0 \leq y \leq 2 \), and \( 0 \leq z \leq 3 \).

#### Solution:

\[
\iiint_V xyz \ dV = \int_0^1 \int_0^2 \int_0^3 xyz \ dz \ dy \ dx
\]

1. Integrate with respect to \( z \):

\[
\int_0^3 xyz \ dz = xyz \left[ z \right]_0^3 = 3xyz
\]

2. Integrate with respect to \( y \):

\[
\int_0^2 3xyz \ dy = 3x \int_0^2 y \ dy = 3x \left[ \frac{y^2}{2} \right]_0^2 = 3x \cdot 2 = 6x
\]

3. Integrate with respect to \( x \):

\[
\int_0^1 6x \ dx = \left[ 3x^2 \right]_0^1 = 3
\]

Thus, the value of the integral is \( 3 \).

---

## Applications of Multiple Integration
1. **Volume Calculation**: Computing the volume under a surface or within a region.
2. **Mass and Density**: Finding mass when density varies over a region.
3. **Centroids and Moments of Inertia**: Used in mechanics and structural engineering.

---

## Practice Problems
1. Evaluate:

    \[
    \iint_R x^2y \ dA, \quad R = [0, 1] \times [0, 2].
    \]

2. Compute:

    \[
    \iiint_V x + y + z \ dV, \quad V: 0 \leq x \leq 1, 0 \leq y \leq 1, 0 \leq z \leq 2.
    \]

3. Solve for the volume of the region bounded by \( x^2 + y^2 \leq 1 \) and \( 0 \leq z \leq 3 \).

---

### Example (Python Code):
```python
from scipy.integrate import dblquad

# Double integral of f(x, y) = x^2 + y^2 over x: [0, 1], y: [0, 1]
def integrand(x, y):
    return x**2 + y**2

result, _ = dblquad(integrand, 0, 1, lambda x: 0, lambda x: 1)
print("Double Integral Result:", result)
```

---


For further reading, check the [documentation](https://mathworld.wolfram.com) or consult advanced calculus textbooks.

