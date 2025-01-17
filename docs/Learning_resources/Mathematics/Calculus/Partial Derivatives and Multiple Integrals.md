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

### Definition
The double integral of \( f(x, y) \) over a region \( R \) is:

\[
\iint_R f(x, y) \, dx \, dy
\]

The triple integral extends this to 3D:

\[
\iiint_R f(x, y, z) \, dx \, dy \, dz
\]

### Real-World Example
- **Double Integral**: Calculate the area under a surface.
- **Triple Integral**: Calculate the volume of a solid.

### Example (Python Code):
```python
from scipy.integrate import dblquad

# Double integral of f(x, y) = x^2 + y^2 over x: [0, 1], y: [0, 1]
def integrand(x, y):
    return x**2 + y**2

result, _ = dblquad(integrand, 0, 1, lambda x: 0, lambda x: 1)
print("Double Integral Result:", result)
```