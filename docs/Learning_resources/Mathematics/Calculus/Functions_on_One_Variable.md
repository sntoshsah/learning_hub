# Calculus: Functions of One Variable

## 1. Functions and Its Representations

### Definition:
A function is a relation between a set of inputs (domain) and a set of possible outputs (range) such that each input is related to exactly one output.

### Representations of Functions:
1. **Algebraic Representation**:
   A function can be expressed as a formula, e.g., \( f(x) = x^2 + 3x + 5 \).

2. **Graphical Representation**:
   A graph of a function shows the relationship between \( x \) (input) and \( f(x) \) (output).

3. **Tabular Representation**:
   A table can list specific \( x \)-values and their corresponding \( f(x) \)-values.

4. **Verbal Description**:
   A function can also be described in words, e.g., "A function that squares a number and adds 2."

### Example:
#### Algebraic Representation:
\[ f(x) = x^2 - 4 \]

#### Graphical Representation:
Plot the points for \( x = -2, -1, 0, 1, 2 \):

| \( x \)  | \( f(x) \) |
|-------|---------|
| -2    | 0       |
| -1    | -3      |
| 0     | -4      |
| 1     | -3      |
| 2     | 0       |

#### Verbal Representation:
"A function that takes a number, squares it, and subtracts 4."

---

## 2. Linear Mathematical Model

### Definition:
A linear mathematical model represents a relationship between variables that can be expressed in the form \( y = mx + b \), where:

- \( m \): Slope of the line
- \( b \): Y-intercept

### Key Features:
- The graph is a straight line.
- Slope \( m \): Describes the rate of change.
- Y-intercept \( b \): The value of \( y \) when \( x = 0 \).

### Example:
#### Problem:
Create a linear model for a car traveling at a constant speed of 60 km/h.

#### Solution:
Let \( x \) represent time (in hours) and \( y \) represent the distance traveled (in km).

The relationship can be modeled as:
\[ y = 60x \]

#### Graphical Representation:
If \( x = 1, 2, 3 \):

| \( x \) (Time in hours) | \( y \) (Distance in km) |
|--------------------------|--------------------------|
| 1                        | 60                       |
| 2                        | 120                      |
| 3                        | 180                      |

---

## 3. Combinations of Functions

### Definition:
Functions can be combined using arithmetic operations (addition, subtraction, multiplication, and division) or composition.

### Arithmetic Combinations:
1. **Addition**: \( (f + g)(x) = f(x) + g(x) \)
2. **Subtraction**: \( (f - g)(x) = f(x) - g(x) \)
3. **Multiplication**: \( (f \cdot g)(x) = f(x) \cdot g(x) \)
4. **Division**: \( \left( \frac{f}{g} \right)(x) = \frac{f(x)}{g(x)}, g(x) \neq 0 \)

### Composition of Functions:
The composition \( (f \circ g)(x) \) is defined as:

\[ (f \circ g)(x) = f(g(x)) \]

### Example:
#### Given Functions:
\( f(x) = 2x + 3 \) and \( g(x) = x^2 \).

#### Combinations:
1. **Addition**: \( (f + g)(x) = (2x + 3) + (x^2) = x^2 + 2x + 3 \)
2. **Subtraction**: \( (f - g)(x) = (2x + 3) - (x^2) = -x^2 + 2x + 3 \)
3. **Multiplication**: \( (f \cdot g)(x) = (2x + 3)(x^2) = 2x^3 + 3x^2 \)
4. **Composition**:
   \[ (f \circ g)(x) = f(g(x)) = f(x^2) = 2(x^2) + 3 = 2x^2 + 3 \]

---

## 4. Rational, Trigonometric, Exponential, and Logarithmic Functions

### Rational Functions:
A rational function is the ratio of two polynomials:

\[ R(x) = \frac{P(x)}{Q(x)} \]

where \( Q(x) \neq 0 \).

#### Example:
\[ R(x) = \frac{x^2 - 1}{x - 2} \]

#### Key Points:
- The domain excludes values that make \( Q(x) = 0 \).

---

### Trigonometric Functions:
These include sine, cosine, tangent, etc., and are periodic in nature.

#### Example:
\( f(x) = \sin(x) \)
- Domain: \( (-\infty, \infty) \)
- Range: \( [-1, 1] \)

#### Key Properties:
- Periodicity: \( \sin(x + 2\pi) = \sin(x) \)
- Symmetry: \( \sin(-x) = -\sin(x) \)

---

### Exponential Functions:
An exponential function has the form:

\[ f(x) = a \cdot b^x \]

where \( b > 0 \) and \( b \neq 1 \).

#### Example:
\( f(x) = 2^x \)

#### Key Points:
- Growth: If \( b > 1 \).
- Decay: If \( 0 < b < 1 \).

---

### Logarithmic Functions:
A logarithmic function is the inverse of an exponential function:

\[ f(x) = \log_b(x) \]

where \( b > 0 \) and \( b \neq 1 \).

#### Example:
\( f(x) = \log_2(x) \)

#### Key Points:

- Domain: \( (0, \infty) \)
- Range: \( (-\infty, \infty) \)
