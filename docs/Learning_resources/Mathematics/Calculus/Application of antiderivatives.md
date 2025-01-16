# Applications of Antiderivatives

## 1. Area Between Two Curves

### Intuition
The area between two curves \( f(x) \) and \( g(x) \) over an interval \([a, b]\) is given by:

\[
\text{Area} = \int_a^b |f(x) - g(x)| \, dx.
\]

If \( f(x) \geq g(x) \) on \([a, b]\):

\[
\text{Area} = \int_a^b (f(x) - g(x)) \, dx.
\]

### Example
Find the area between \( f(x) = x^2 \) and \( g(x) = x \) over \([0, 1]\).

\[
\text{Area} = \int_0^1 (x - x^2) \, dx = \left[\frac{x^2}{2} - \frac{x^3}{3}\right]_0^1 = \frac{1}{2} - \frac{1}{3} = \frac{1}{6}.
\]

### Python Visualization
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define the functions
def f(x):
    return x

def g(x):
    return x**2

# Compute the area
area, _ = quad(lambda x: f(x) - g(x), 0, 1)

# Plot
x = np.linspace(0, 1, 500)
plt.plot(x, f(x), label='f(x) = x')
plt.plot(x, g(x), label='g(x) = x^2')
plt.fill_between(x, g(x), f(x), alpha=0.2, label=f'Area = {area:.2f}')
plt.legend()
plt.title('Area Between Two Curves')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
```

---

## 2. Volumes of Cylindrical Shells

### Intuition
The volume of a solid of revolution using the shell method is given by:

\[
V = 2\pi \int_a^b \text{(radius)} \cdot \text{(height)} \, dx.
\]

Here:

- Radius is the distance from the axis of rotation.
- Height is the function value.

### Example
Find the volume of the solid obtained by rotating \( y = x^2 \) about the y-axis over \([0, 1]\):

\[
V = 2\pi \int_0^1 x \cdot x^2 \, dx = 2\pi \int_0^1 x^3 \, dx = 2\pi \left[\frac{x^4}{4}\right]_0^1 = \frac{\pi}{2}.
\]

### Python Visualization
```python
# Define the function
def height(x):
    return x**2

# Compute volume
volume, _ = quad(lambda x: 2 * np.pi * x * height(x), 0, 1)

# Plot
x = np.linspace(0, 1, 500)
y = height(x)
plt.plot(x, y, label='y = x^2')
plt.fill_between(x, 0, y, alpha=0.2, label=f'Volume = {volume:.2f}')
plt.legend()
plt.title('Volume by Cylindrical Shells')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
```

---

## 3. Approximate Integration

### Intuition
Approximate integration techniques, like the Trapezoidal Rule and Simpson’s Rule, estimate definite integrals by dividing the interval into smaller subintervals.

### Example: Trapezoidal Rule
\[
\int_a^b f(x) \, dx \approx \frac{\Delta x}{2} \left[f(x_0) + 2f(x_1) + \cdots + f(x_n)\right].
\]

### Python Visualization
```python
# Define function
def f(x):
    return x**2

# Trapezoidal approximation
a, b = 0, 1
n = 10
x = np.linspace(a, b, n+1)
y = f(x)
delta_x = (b - a) / n
trapezoidal_approx = (delta_x / 2) * (y[0] + 2 * sum(y[1:-1]) + y[-1])

# Plot
x_dense = np.linspace(a, b, 500)
y_dense = f(x_dense)
plt.plot(x_dense, y_dense, label='f(x) = x^2')
plt.bar(x[:-1], y[:-1], width=delta_x, align='edge', alpha=0.3, label='Trapezoids')
plt.title(f'Trapezoidal Approximation: {trapezoidal_approx:.2f}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()
```

---

## 4. Arc Length

### Intuition
The length of a curve \( y = f(x) \) over \([a, b]\) is given by:

\[
L = \int_a^b \sqrt{1 + \left(\frac{dy}{dx}\right)^2} \, dx.
\]

### Example
Find the arc length of \( y = \sqrt{x} \) over \([0, 1]\):

\[
L = \int_0^1 \sqrt{1 + \left(\frac{1}{2\sqrt{x}}\right)^2} \, dx.
\]

### Python Visualization
```python
# Define function and its derivative
def f(x):
    return np.sqrt(x)

def f_prime(x):
    return 1 / (2 * np.sqrt(x))

# Compute arc length
arc_length, _ = quad(lambda x: np.sqrt(1 + f_prime(x)**2), 0.01, 1)  # Avoid division by zero

# Plot
x = np.linspace(0, 1, 500)
y = f(x)
plt.plot(x, y, label='y = sqrt(x)')
plt.title(f'Arc Length = {arc_length:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
```

---

## 5. Area of Surface of Revolution

### Intuition
The surface area of a solid obtained by rotating \( y = f(x) \) about the x-axis is:

\[
S = 2\pi \int_a^b f(x) \sqrt{1 + \left(\frac{dy}{dx}\right)^2} \, dx.
\]

### Example
Find the surface area of \( y = x^2 \) rotated about the x-axis over \([0, 1]\).

\[
S = 2\pi \int_0^1 x^2 \sqrt{1 + (2x)^2} \, dx.
\]

### Python Visualization
```python
# Compute surface area
def f(x):
    return x**2

def f_prime(x):
    return 2*x

surface_area, _ = quad(lambda x: 2 * np.pi * f(x) * np.sqrt(1 + f_prime(x)**2), 0, 1)

# Plot
x = np.linspace(0, 1, 500)
y = f(x)
plt.plot(x, y, label='y = x^2')
plt.title(f'Surface Area = {surface_area:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
```