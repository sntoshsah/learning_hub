# Applications of Derivatives

## 1. Curve Sketching

### Intuition
Curve sketching involves using the derivative to analyze and draw the graph of a function. The first and second derivatives provide crucial information:

- **First Derivative (f'(x))**: Indicates where the function is increasing or decreasing and helps find critical points (local maxima and minima).
- **Second Derivative (f''(x))**: Indicates concavity (whether the curve bends upwards or downwards) and helps locate inflection points.

### Steps for Curve Sketching
1. Find the domain of the function.
2. Determine the critical points by solving \( f'(x) = 0 \) or \( f'(x) \) undefined.
3. Analyze intervals of increase/decrease using the sign of \( f'(x) \).
4. Find the concavity and inflection points using \( f''(x) \).
5. Plot key points and use the information to sketch the graph.

### Example
Sketch the curve of \( f(x) = x^3 - 3x^2 + 4 \).

### Python Visualization
```python
import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivatives
def f(x):
    return x**3 - 3*x**2 + 4

def f_prime(x):
    return 3*x**2 - 6*x

def f_double_prime(x):
    return 6*x - 6

# Generate x values
x = np.linspace(-1, 3, 500)
y = f(x)

# Critical points
critical_points = [0, 2]
inflection_point = [1]

# Plot the function
plt.plot(x, y, label='f(x) = x^3 - 3x^2 + 4')
plt.scatter(critical_points, f(np.array(critical_points)), color='red', label='Critical Points')
plt.scatter(inflection_point, f(np.array(inflection_point)), color='green', label='Inflection Point')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()
plt.title('Curve Sketching of f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.show()
```

---

## 2. Optimization Problems

### Intuition
Optimization involves finding the maximum or minimum values of a function. These problems often occur in real-world scenarios like minimizing costs or maximizing profit.

### Steps for Solving Optimization Problems
1. Define the function to optimize.
2. Identify constraints and express them mathematically.
3. Use derivatives to find critical points.
4. Analyze critical points and endpoints to determine the optimal solution.

### Example
Find the dimensions of a rectangle with a perimeter of 20 units that maximizes its area.

#### Solution
1. Let the dimensions be \( x \) and \( y \), with \( 2x + 2y = 20 \).
2. Express the area: \( A = x \cdot y \).
3. Substitute \( y = 10 - x \): \( A(x) = x(10 - x) = 10x - x^2 \).
4. Maximize \( A(x) \) by finding \( A'(x) = 10 - 2x \).

### Python Visualization
```python
x = np.linspace(0, 10, 500)
def area(x):
    return x * (10 - x)

y = area(x)

plt.plot(x, y, label='Area(x) = 10x - x^2')
plt.axvline(5, color='red', linestyle='--', label='Max Area at x=5')
plt.title('Optimization: Maximize Area')
plt.xlabel('Width (x)')
plt.ylabel('Area')
plt.legend()
plt.grid()
plt.show()
```

---

## 3. Newton's Method

### Intuition
Newton's Method is an iterative numerical technique to approximate roots of a function \( f(x) \). Starting from an initial guess \( x_0 \), the method improves the guess using:

\[
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
\]

The process continues until the approximation converges to a root.

### Example
Approximate the root of \( f(x) = x^2 - 2 \) using Newton's Method.

### Python Visualization
```python
# Define the function and its derivative
def f(x):
    return x**2 - 2

def f_prime(x):
    return 2*x

# Newton's Method
def newtons_method(f, f_prime, x0, tolerance=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        x_new = x - f(x) / f_prime(x)
        if abs(x_new - x) < tolerance:
            break
        x = x_new
    return x

# Initial guess
x0 = 1.5
root = newtons_method(f, f_prime, x0)

# Plot the function and tangent lines
x = np.linspace(0, 2, 500)
y = f(x)

plt.plot(x, y, label='f(x) = x^2 - 2')
plt.axhline(0, color='black', linestyle='--')
plt.scatter([root], [f(root)], color='red', label=f'Root ~ {root:.4f}')
plt.legend()
plt.title("Newton's Method")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.show()
```