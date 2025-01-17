# Calculus Concepts: Integration

## 1. Rectilinear Motion, Area, and Distances

### Intuition
Integration helps calculate quantities like area under curves, total distance traveled, or the displacement of an object in rectilinear motion. For motion:

- The position \( s(t) \) is the integral of velocity \( v(t) \).
- The total distance is the integral of the absolute value of velocity.

### Example
A particle's velocity is \( v(t) = t^2 \) m/s. Find the total distance traveled from \( t=0 \) to \( t=2 \).

\[
\text{Distance} = \int_0^2 |v(t)| \, dt = \int_0^2 t^2 \, dt
\]

### Python Visualization
```python
import numpy as np
import matplotlib.pyplot as plt

# Define velocity function
def v(t):
    return t**2

# Time range
t = np.linspace(0, 2, 500)
distance = np.cumsum(v(t) * (t[1] - t[0]))

# Plot
plt.plot(t, v(t), label='v(t) = t^2')
plt.fill_between(t, v(t), alpha=0.2, label='Distance Traveled')
plt.title('Rectilinear Motion: Velocity and Distance')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid()
plt.show()
```

---

## 2. The Definite Integral

### Intuition
The definite integral calculates the net area between the curve \( f(x) \) and the x-axis over an interval \([a, b]\). If \( f(x) \) is above the axis, the area is positive; if below, it is negative.

### Mathematical Definition

\[
\int_a^b f(x) \, dx = F(b) - F(a),
\]

where \( F(x) \) is the antiderivative of \( f(x) \).

### Example
Find \( \int_0^3 (x^2 + 1) \, dx \).

\[
\int_0^3 (x^2 + 1) \, dx = \left[ \frac{x^3}{3} + x \right]_0^3 = \left(\frac{27}{3} + 3\right) - (0 + 0) = 12
\]

### Python Visualization
```python
from scipy.integrate import quad

# Define function
def f(x):
    return x**2 + 1

# Compute definite integral
area, _ = quad(f, 0, 3)

# Plot
x = np.linspace(0, 3, 500)
y = f(x)
plt.plot(x, y, label='f(x) = x^2 + 1')
plt.fill_between(x, y, alpha=0.2, label=f'Area = {area:.2f}')
plt.title('Definite Integral')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()
```

---

## 3. The Fundamental Theorem of Calculus

### Intuition
This theorem bridges differentiation and integration:

1. If \( F(x) \) is the antiderivative of \( f(x) \), then:

    \[
    \int_a^b f(x) \, dx = F(b) - F(a).
    \]

2. The derivative of the integral is the original function:

    \[
    \frac{d}{dx} \int_a^x f(t) \, dt = f(x).
    \]

### Example
Verify the fundamental theorem for \( f(x) = x^2 \) over \([0, 2]\):

\[
F(x) = \frac{x^3}{3}, \quad \int_0^2 x^2 \, dx = \frac{2^3}{3} - \frac{0^3}{3} = \frac{8}{3}.
\]

### Python Visualization
```python
# Define function and its antiderivative
def f(x):
    return x**2

def F(x):
    return x**3 / 3

# Plot
x = np.linspace(0, 2, 500)
plt.plot(x, f(x), label='f(x) = x^2')
plt.plot(x, F(x), '--', label='F(x) = x^3 / 3')
plt.title('Fundamental Theorem of Calculus')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
```

---

## 4. Indefinite Integral and the Net Change Theorem

### Intuition
The indefinite integral represents a family of functions, \( \int f(x) \, dx = F(x) + C \), where \( C \) is the constant of integration. The net change theorem states:

\[
\int_a^b f'(x) \, dx = f(b) - f(a),
\]

indicating that integration gives the total change.

### Example
Find the indefinite integral of \( f(x) = 3x^2 \):

\[
\int 3x^2 \, dx = x^3 + C.
\]

### Python Visualization
```python
from sympy import symbols, integrate

# Symbolic computation
x = symbols('x')
expr = 3*x**2
indef_integral = integrate(expr, x)
print(f"Indefinite Integral: {indef_integral} + C")
```

---

## 5. Techniques of Integration

### Intuition
Techniques like substitution, integration by parts, and partial fractions simplify complex integrals.

### Example: Substitution
Evaluate \( \int 2x \sqrt{x^2 + 1} \, dx \):

1. Substitute \( u = x^2 + 1 \), \( du = 2x \, dx \).
2. Integral becomes \( \int \sqrt{u} \, du \).
3. Solve \( \frac{2}{3} u^{3/2} + C \).

### Python Visualization
```python
from sympy import sqrt

# Symbolic computation with substitution
u = symbols('u')
expr_sub = sqrt(u)
sub_integral = integrate(expr_sub, u)
print(f"After Substitution: {sub_integral} + C")
```

---

## 6. Improper Integral

### Intuition
Improper integrals extend the concept of integration to unbounded intervals or functions with infinite discontinuities. They are evaluated as limits.

### Example
Evaluate \( \int_1^\infty \frac{1}{x^2} \, dx \):

\[
\int_1^\infty \frac{1}{x^2} \, dx = \lim_{b \to \infty} \left[-\frac{1}{x}\right]_1^b = \lim_{b \to \infty} \left(-\frac{1}{b} + 1\right) = 1.
\]

### Python Visualization
```python
# Define the function
def improper_integral(x):
    return 1 / x**2

# Compute numerical approximation
improper_area, _ = quad(improper_integral, 1, np.inf)
print(f"Improper Integral Result: {improper_area}")
```

This file covers key integration concepts with detailed explanations, examples, and Python visualizations. Let me know if you need further adjustments or additions!
