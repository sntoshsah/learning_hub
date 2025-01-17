# Integration Techniques

Integration is a fundamental concept in calculus used to compute areas under curves, solve differential equations, and analyze continuous data. Various techniques of integration are employed based on the nature of the integrand. This document provides a detailed overview of these techniques.

---

## 1. **Basic Integration Rules**

Before diving into advanced techniques, understanding the basic rules is essential:

- **Power Rule:**

\[
\int x^n dx = \frac{x^{n+1}}{n+1} + C, \quad \text{for } n \neq -1
\]

- **Constant Rule:**

\[
\int a \cdot f(x) dx = a \int f(x) dx
\]

- **Sum/Difference Rule:**

\[
\int \big(f(x) \pm g(x)\big) dx = \int f(x) dx \pm \int g(x) dx
\]

- **Exponential and Logarithmic Rules:**

\[
\int e^x dx = e^x + C \quad \text{and} \quad \int \frac{1}{x} dx = \ln|x| + C
\]

---

## 2. **Substitution Method**

The substitution method simplifies integrals by transforming the variable:

- **Steps:**
  1. Identify a substitution \( u = g(x) \).
  2. Compute \( du = g'(x) dx \).
  3. Rewrite the integral in terms of \( u \).
  4. Integrate with respect to \( u \) and substitute back.

- **Example:**

\[
\int x e^{x^2} dx \quad \text{Let } u = x^2, \; du = 2x dx.
\]

\[
\int x e^{x^2} dx = \frac{1}{2} \int e^u du = \frac{1}{2} e^u + C = \frac{1}{2} e^{x^2} + C.
\]

---

## 3. **Integration by Parts**

Integration by parts is based on the product rule of differentiation:

\[
\int u \cdot v' dx = uv - \int u' \cdot v dx.
\]

- **Steps:**
  1. Choose \( u \) and \( dv \) (using the LIATE rule: Logarithmic, Inverse trig, Algebraic, Trigonometric, Exponential).
  2. Differentiate \( u \) to find \( du \) and integrate \( dv \) to find \( v \).
  3. Apply the formula.

- **Example:**

    \[
    \int x e^x dx \quad \text{Choose } u = x, \; dv = e^x dx.
    \]

    \[
    u = x, \; du = dx, \; v = e^x.
    \]
    
    \[
    \int x e^x dx = x e^x - \int e^x dx = x e^x - e^x + C = e^x(x - 1) + C.
    \]

---

## 4. **Trigonometric Integrals**

Trigonometric integrals involve powers of sine, cosine, and other trigonometric functions.

- **Key Strategies:**
  - For \( \sin^m(x) \cos^n(x) \), reduce powers using \( \sin^2(x) + \cos^2(x) = 1 \).
  - Use half-angle identities when appropriate.

- **Example:**

\[
\int \sin^2(x) dx = \int \frac{1 - \cos(2x)}{2} dx = \frac{1}{2} \int dx - \frac{1}{2} \int \cos(2x) dx.
\]

\[
= \frac{x}{2} - \frac{1}{4} \sin(2x) + C.
\]

---

## 5. **Partial Fraction Decomposition**

This method is used for rational functions where the degree of the numerator is less than the degree of the denominator.

- **Steps:**
  1. Factor the denominator.
  2. Decompose into partial fractions.
  3. Integrate each term.

- **Example:**

\[
\int \frac{1}{x^2 - 1} dx = \int \frac{1}{(x - 1)(x + 1)} dx.
\]

\[
\frac{1}{x^2 - 1} = \frac{A}{x - 1} + \frac{B}{x + 1} \quad \text{Solve for } A, B.
\]

\[
\int \frac{1}{x^2 - 1} dx = \int \frac{1}{2(x - 1)} dx - \int \frac{1}{2(x + 1)} dx.
\]

\[
= \frac{1}{2} \ln|x - 1| - \frac{1}{2} \ln|x + 1| + C.
\]

---

## 6. **Trigonometric Substitution**

This technique is used for integrals involving \( \sqrt{a^2 - x^2} \), \( \sqrt{a^2 + x^2} \), or \( \sqrt{x^2 - a^2} \).

- **Substitutions:**
  - For \( \sqrt{a^2 - x^2} \), use \( x = a \sin\theta \).
  - For \( \sqrt{a^2 + x^2} \), use \( x = a \tan\theta \).
  - For \( \sqrt{x^2 - a^2} \), use \( x = a \sec\theta \).

- **Example:**

\[
\int \frac{dx}{\sqrt{a^2 - x^2}} \quad \text{Let } x = a \sin\theta, \; dx = a \cos\theta d\theta.
\]

\[
= \int \frac{a \cos\theta d\theta}{a \cos\theta} = \int d\theta = \theta + C = \arcsin\frac{x}{a} + C.
\]

---

## 7. **Improper Integrals**

Improper integrals arise when the limits of integration are infinite or the integrand has a singularity.

- **Steps:**
  1. Replace the problematic limit or singularity with a variable.
  2. Take the limit as the variable approaches the problematic value.

- **Example:**

\[
\int_{1}^{\infty} \frac{1}{x^2} dx = \lim_{b \to \infty} \int_{1}^{b} \frac{1}{x^2} dx.
\]

\[
= \lim_{b \to \infty} \big[-\frac{1}{x}\big]_1^b = \lim_{b \to \infty} \big(-\frac{1}{b} + 1\big) = 1.
\]

---

## Conclusion

Integration techniques provide the tools to solve a wide range of problems in mathematics, physics, and engineering. Mastery of these methods comes with practice and understanding of when to apply each technique effectively.
