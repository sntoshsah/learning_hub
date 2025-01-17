# Infinite Sequences and Series


## 1. Infinite Sequence and Series

### Definition
An **infinite sequence** is an ordered list of elements, typically numbers, that extends indefinitely. Each term in the sequence is usually defined by a formula.

A **series** is the sum of the terms of a sequence. An **infinite series** is the sum of an infinite sequence:

\[
S = a_1 + a_2 + a_3 + \dots + a_n + \dots
\]

### Explanation and Intuition
- **Infinite Sequence**: Think of a sequence as an endless list of numbers arranged in a specific order. For example, \( \{1, 1/2, 1/3, 1/4, \dots\} \) is an infinite sequence where each term is \( 1/n \).
- **Infinite Series**: When we add the terms of a sequence, we form a series. For instance, the series for \( \{1, 1/2, 1/3, \dots\} \) is:

\[
S = 1 + \frac{1}{2} + \frac{1}{3} + \dots
\]

---

## 2. Convergence Test of Infinite Series

### Convergence and Divergence
A series converges if the sum of its terms approaches a finite value as the number of terms goes to infinity. Otherwise, it diverges.

### Tests for Convergence

#### a) **Geometric Series Test**
A geometric series has the form:

\[
S = a + ar + ar^2 + ar^3 + \dots
\]

- Converges if \( |r| < 1 \).
- Diverges if \( |r| \geq 1 \).

#### Example (Python Code):
```python
import numpy as np
import matplotlib.pyplot as plt

# Geometric series example
def geometric_series(a, r, n):
    terms = [a * (r ** i) for i in range(n)]
    return terms

# Parameters
a, r, n = 1, 0.5, 20
terms = geometric_series(a, r, n)
cumsum = np.cumsum(terms)

plt.plot(cumsum, marker='o', label='Cumulative Sum')
plt.axhline(y=sum(terms), color='r', linestyle='--', label='Convergent Value')
plt.xlabel('Number of Terms')
plt.ylabel('Sum')
plt.title('Convergence of Geometric Series')
plt.legend()
plt.show()
```

#### b) **p-Series Test**
A p-series has the form:

\[
S = \sum_{n=1}^{\infty} \frac{1}{n^p}
\]

- Converges if \( p > 1 \).
- Diverges if \( p \leq 1 \).

#### Example (Python Code):
```python
# p-series example
def p_series(p, n):
    terms = [1 / (i ** p) for i in range(1, n + 1)]
    return terms

# Parameters
p, n = 2, 50
terms = p_series(p, n)
cumsum = np.cumsum(terms)

plt.plot(cumsum, marker='o', label=f'p = {p}')
plt.xlabel('Number of Terms')
plt.ylabel('Sum')
plt.title('Convergence of p-Series')
plt.legend()
plt.show()
```

#### c) **Ratio Test**
For a series \( \sum a_n \), if:

\[
L = \lim_{n \to \infty} \left| \frac{a_{n+1}}{a_n} \right|
\]

- Converges if \( L < 1 \).
- Diverges if \( L > 1 \).
- Inconclusive if \( L = 1 \).

#### Example:
The factorial series \( \sum \frac{1}{n!} \) converges because the ratio test gives \( L = 0 \).

---

## 3. Power Series, Taylor, and Maclaurin Series

### Power Series
A power series is a series of the form:

\[
S(x) = \sum_{n=0}^{\infty} c_n (x - a)^n
\]

- \( a \): Center of the series.
- \( c_n \): Coefficients.

#### Example (Python Code):
```python
# Power series example
x = np.linspace(-2, 2, 100)
a, c = 0, [1, -1/2, 1/3, -1/4]

# Compute power series
def power_series(x, c, a):
    return sum(c[i] * (x - a) ** i for i in range(len(c)))

y = power_series(x, c, a)

plt.plot(x, y, label='Power Series')
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('S(x)')
plt.title('Visualization of Power Series')
plt.legend()
plt.show()
```

### Taylor and Maclaurin Series

- **Taylor Series**: Expands a function \( f(x) \) around a point \( a \):

\[
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x - a)^n
\]

- **Maclaurin Series**: Special case of the Taylor series where \( a = 0 \).

#### Example (Maclaurin Series for \( e^x \)):
```python
# Maclaurin series for e^x
from math import factorial

def maclaurin_exponential(x, terms):
    return sum((x ** n) / factorial(n) for n in range(terms))

x_vals = np.linspace(-2, 2, 100)
y_vals = [maclaurin_exponential(x, 10) for x in x_vals]

plt.plot(x_vals, np.exp(x_vals), label='e^x (Exact)')
plt.plot(x_vals, y_vals, '--', label='Maclaurin Series (Approximation)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Maclaurin Series Approximation for e^x')
plt.legend()
plt.show()
```