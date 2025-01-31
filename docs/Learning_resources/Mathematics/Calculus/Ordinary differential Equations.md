# Ordinary Differential Equations

## 1. Review of Ordinary Differential Equations (ODE)

### Intuition
An Ordinary Differential Equation (ODE) relates a function \( y(x) \) to its derivatives. The general form of an ODE is:

\[
F(x, y, y', y'', \dots, y^{(n)}) = 0.
\]

ODEs are classified based on:

- **Order:** The highest derivative present.
- **Linearity:** Whether the equation can be written as a linear combination of the dependent variable and its derivatives.

### Example

\[
y' + y = 0 \quad \text{(First-order linear ODE)}.
\]

The solution is \( y = Ce^{-x} \), where \( C \) is a constant.

### Python Visualization
```python
import numpy as np
import matplotlib.pyplot as plt

# Define the solution
def y(x, C):
    return C * np.exp(-x)

# Plot
x = np.linspace(0, 5, 500)
C = 1
plt.plot(x, y(x, C), label='y = Ce^{-x}')
plt.title('Solution of y' + y = 0')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
```

---

## 2. Linear Equations

### Intuition

A first-order linear ODE has the form:

\[
y' + P(x)y = Q(x).
\]

The solution is obtained using an **integrating factor**:

\[
\mu(x) = e^{\int P(x) \, dx}, \quad y = \frac{1}{\mu(x)} \int \mu(x)Q(x) \, dx.
\]

### Example
Solve \( y' + y = x \):

\[
P(x) = 1, \quad Q(x) = x, \quad \mu(x) = e^x.
\]

\[
y = e^{-x} \int e^x x \, dx = e^{-x}(x e^x - e^x) + C = x - 1 + Ce^{-x}.
\]

### Python Visualization
```python
from sympy import symbols, Function, Eq, exp, dsolve

# Define symbols
x = symbols('x')
y = Function('y')

# Define ODE
otde = Eq(y(x).diff(x) + y(x), x)

# Solve ODE
solution = dsolve(otde)
print(f"Solution: {solution}")

# Plot
C = 1
x_vals = np.linspace(0, 5, 500)
y_vals = x_vals - 1 + C * np.exp(-x_vals)
plt.plot(x_vals, y_vals, label='y = x - 1 + Ce^{-x}')
plt.title('Solution of y' + y = x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
```

---

## 3. Second-Order Linear Equations

### Intuition
A second-order linear ODE has the form:

\[
ay'' + by' + cy = 0.
\]

The solution depends on the roots of the characteristic equation \( ar^2 + br + c = 0 \):

1. **Distinct real roots:** \( y = C_1 e^{r_1x} + C_2 e^{r_2x} \).
2. **Repeated root:** \( y = (C_1 + C_2x)e^{r_1x} \).
3. **Complex roots:** \( y = e^{\alpha x}(C_1 \cos \beta x + C_2 \sin \beta x) \).

### Example
Solve \( y'' - 3y' + 2y = 0 \):

\[
\text{Characteristic equation: } r^2 - 3r + 2 = 0, \quad r = 1, 2.
\]

\[
y = C_1 e^x + C_2 e^{2x}.
\]

### Python Visualization
```python
# Define ODE
otde2 = Eq(y(x).diff(x, 2) - 3*y(x).diff(x) + 2*y(x), 0)

# Solve ODE
solution2 = dsolve(otde2)
print(f"Solution: {solution2}")

# Plot
C1, C2 = 1, 1
x_vals = np.linspace(0, 5, 500)
y_vals = C1 * np.exp(x_vals) + C2 * np.exp(2 * x_vals)
plt.plot(x_vals, y_vals, label='y = C1 e^x + C2 e^{2x}')
plt.title('Solution of y'' - 3y' + 2y = 0')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
```

---

## 4. Non-Homogeneous Linear Equations

### Intuition
A second-order non-homogeneous ODE has the form:

\[
ay'' + by' + cy = G(x).
\]

The solution is:

\[
y = y_h + y_p
\]

where:

- \( y_h \): Solution of the homogeneous equation.
- \( y_p \): Particular solution.

### Example
Solve \( y'' - y = e^x \):

\[
y_h = C_1 e^x + C_2 e^{-x}, \quad y_p = \frac{1}{2}x e^x.
\]

\[
y = C_1 e^x + C_2 e^{-x} + \frac{1}{2}x e^x.
\]

### Python Visualization
```python
# Define ODE
otde3 = Eq(y(x).diff(x, 2) - y(x), exp(x))

# Solve ODE
solution3 = dsolve(otde3)
print(f"Solution: {solution3}")

# Plot
C1, C2 = 1, 1
x_vals = np.linspace(0, 5, 500)
y_vals = C1 * np.exp(x_vals) + C2 * np.exp(-x_vals) + 0.5 * x_vals * np.exp(x_vals)
plt.plot(x_vals, y_vals, label='y = C1 e^x + C2 e^{-x} + 0.5x e^x')
plt.title('Solution of y'' - y = e^x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
```
