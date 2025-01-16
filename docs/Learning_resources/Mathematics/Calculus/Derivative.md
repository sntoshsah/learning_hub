# Calculus Concepts: Derivatives

## 1. Tangent and Velocity

### Intuition
The concept of a derivative originates from the need to find the slope of a tangent line to a curve at a point. In physics, this translates to determining the instantaneous velocity of an object.

The slope of a secant line connecting two points on a curve gives an average rate of change. As the two points move closer, the secant line approaches the tangent line, and the slope approaches the derivative.

### Mathematical Definition
The derivative of a function \( f(x) \) at a point \( x = a \) is given by:

\[
\lim_{h \to 0} \frac{f(a + h) - f(a)}{h}
\]

### Example
Find the derivative of \( f(x) = x^2 \) at \( x = 2 \).

\[
f'(2) = \lim_{h \to 0} \frac{(2 + h)^2 - 2^2}{h} = \lim_{h \to 0} \frac{4 + 4h + h^2 - 4}{h} = \lim_{h \to 0} (4 + h) = 4
\]

### Python Visualization
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4, 100)
f = x**2

def tangent_line(x, a):
    slope = 2 * a
    return slope * (x - a) + a**2

x_tangent = 2
y_tangent = tangent_line(x, x_tangent)

plt.plot(x, f, label='f(x) = x^2')
plt.plot(x, y_tangent, '--', label=f'Tangent at x={x_tangent}')
plt.scatter([x_tangent], [x_tangent**2], color='red')
plt.legend()
plt.title('Tangent and Velocity')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.show()
```

---

## 2. Rate of Change

### Intuition
The rate of change measures how a quantity changes with respect to another. For example, speed is the rate of change of distance with respect to time.

### Example
The rate of change of \( f(x) = x^3 \) from \( x = 1 \) to \( x = 2 \) is:

\[
\text{Average rate of change} = \frac{f(2) - f(1)}{2 - 1} = \frac{8 - 1}{1} = 7
\]

### Python Visualization
```python
x = np.linspace(0, 3, 100)
f = x**3

x1, x2 = 1, 2
y1, y2 = x1**3, x2**3

plt.plot(x, f, label='f(x) = x^3')
plt.plot([x1, x2], [y1, y2], 'o-', label='Secant Line')
plt.legend()
plt.title('Rate of Change')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.show()
```

---

## 3. Derivative as a Function

### Intuition
The derivative itself can be treated as a function, \( f'(x) \), that gives the slope of the tangent line to \( f(x) \) at any point \( x \).

### Example
For \( f(x) = x^3 \),

\[
f'(x) = 3x^2
\]

### Python Visualization
```python
x = np.linspace(-2, 2, 100)
f = x**3
f_prime = 3 * x**2

plt.plot(x, f, label='f(x) = x^3')
plt.plot(x, f_prime, label="f'(x) = 3x^2")
plt.legend()
plt.title('Derivative as a Function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
```

---

## 4. Review of Derivative

### Intuition
Derivatives are foundational to calculus, enabling us to:
- Calculate slopes of tangent lines.
- Analyze rates of change.
- Solve real-world problems in physics, biology, and economics.

### Key Rules
- Power Rule: \( \frac{d}{dx} [x^n] = nx^{n-1} \)
- Sum Rule: \( \frac{d}{dx} [f(x) + g(x)] = f'(x) + g'(x) \)
- Product Rule: \( \frac{d}{dx} [f(x)g(x)] = f'(x)g(x) + f(x)g'(x) \)
- Quotient Rule: \( \frac{d}{dx} \left[ \frac{f(x)}{g(x)} \right] = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2} \)
- Chain Rule: \( \frac{d}{dx} [f(g(x))] = f'(g(x))g'(x) \)

---

## 5. Mean Value Theorem

### Intuition
If \( f(x) \) is continuous and differentiable on \([a, b]\), there exists a point \( c \) in \((a, b)\) such that:

\[
f'(c) = \frac{f(b) - f(a)}{b - a}
\]

### Example
For \( f(x) = x^2 \) on \([1, 3]\):

\[
\text{Average slope} = \frac{f(3) - f(1)}{3 - 1} = \frac{9 - 1}{2} = 4
\]

The derivative \( f'(x) = 2x \) satisfies \( f'(c) = 4 \) at \( c = 2 \).

### Python Visualization
```python
x = np.linspace(0, 4, 100)
f = x**2

x1, x2 = 1, 3
y1, y2 = x1**2, x2**2

def mean_value(x):
    return 4 * (x - 2) + 4

x_mvt = 2
y_mvt = mean_value(x)

plt.plot(x, f, label='f(x) = x^2')
plt.plot([x1, x2], [y1, y2], 'o-', label='Secant Line')
plt.plot(x, y_mvt, '--', label='Tangent at x=2')
plt.legend()
plt.title('Mean Value Theorem')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.show()
```

---

## 6. Indeterminate Forms and L'Hopital's Rule

### Intuition
Indeterminate forms like \( \frac{0}{0} \) or \( \frac{\infty}{\infty} \) arise in calculus. L'Hopital's Rule provides a method to evaluate such limits:

\[
\lim_{x \to c} \frac{f(x)}{g(x)} = \lim_{x \to c} \frac{f'(x)}{g'(x)}, \text{ if the limit exists.}
\]

### Example
Evaluate \( \lim_{x \to 0} \frac{\sin x}{x} \).

\[
\text{Using L'Hopital's Rule: } \lim_{x \to 0} \frac{\sin x}{x} = \lim_{x \to 0} \frac{\cos x}{1} = 1
\]

### Python Visualization

```python
x = np.linspace(-1, 1, 100)
y = np.sin(x) / x

def safe_division(x):
    return np.where(x == 0, 1, np.sin(x) / x)

y_safe = safe_division(x)

plt.plot(x, y_safe, label='sin(x)/x')
plt.axhline(1, color='red', linestyle='--', label='y=1')
plt.legend()
plt.title("L'Hopital's Rule")
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
```