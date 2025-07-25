# Asymptotic Notations and Analysis

## Introduction to Asymptotic Analysis

Asymptotic analysis is a method of describing the **limiting behavior** of algorithms as the input size grows towards infinity. It provides fundamental metrics for comparing algorithm efficiency independent of machine-specific constants.

## Types of Asymptotic Notations

### 1. Big-O Notation (O) - Upper Bound

**Definition**: 

```
f(n) = O(g(n)) if there exist positive constants c and n₀ such that:
0 ≤ f(n) ≤ c·g(n) for all n ≥ n₀
```

**Interpretation**: 

- "f(n) grows no faster than g(n)"
- Worst-case scenario for algorithm complexity

**Example**:
```
3n² + 2n + 1 = O(n²) with c=6 and n₀=1
Because 3n² + 2n + 1 ≤ 6n² for all n≥1
```

### 2. Omega Notation (Ω) - Lower Bound

**Definition**:
```
f(n) = Ω(g(n)) if there exist positive constants c and n₀ such that:
0 ≤ c·g(n) ≤ f(n) for all n ≥ n₀
```

**Interpretation**:

- "f(n) grows at least as fast as g(n)"
- Best-case scenario for algorithm complexity

**Example**:
```
3n² + 2n + 1 = Ω(n²) with c=1 and n₀=0
Because n² ≤ 3n² + 2n + 1 for all n≥0
```

### 3. Theta Notation (Θ) - Tight Bound

**Definition**:
```
f(n) = Θ(g(n)) if there exist positive constants c₁, c₂ and n₀ such that:
0 ≤ c₁·g(n) ≤ f(n) ≤ c₂·g(n) for all n ≥ n₀
```

**Interpretation**:

- "f(n) grows exactly as g(n)"
- Both upper and lower bounds match

**Example**:
```
3n² + 2n + 1 = Θ(n²) with c₁=1, c₂=6, n₀=1
Because n² ≤ 3n² + 2n + 1 ≤ 6n² for all n≥1
```

### 4. Little-o Notation (o) - Strict Upper Bound

**Definition**:
```
f(n) = o(g(n)) if for any positive constant c, there exists n₀ such that:
0 ≤ f(n) < c·g(n) for all n ≥ n₀
```

**Interpretation**:

- "f(n) grows strictly slower than g(n)"
- Stronger statement than Big-O

### 5. Little-omega Notation (ω) - Strict Lower Bound

**Definition**:
```
f(n) = ω(g(n)) if for any positive constant c, there exists n₀ such that:
0 ≤ c·g(n) < f(n) for all n ≥ n₀
```

**Interpretation**:

- "f(n) grows strictly faster than g(n)"
- Stronger statement than Omega

## Calculating Asymptotic Complexity

### Method 1: Direct Analysis

**Steps**:

1. Identify the dominant term (highest order term)
2. Drop constant coefficients
3. Express using appropriate asymptotic notation

**Example**:
```
T(n) = 5n³ + 2n² + 100n + 1000
Dominant term: 5n³
Drop coefficient: n³
Thus, T(n) = Θ(n³)
```

### Method 2: Limit Comparison

Compute:
```
lim (n→∞) f(n)/g(n)
```

**Results**:

- If limit = constant > 0 → f(n) = Θ(g(n))
- If limit = 0 → f(n) = o(g(n)) and O(g(n))
- If limit = ∞ → f(n) = ω(g(n)) and Ω(g(n))

**Example**:
```
Compare n² and n³:
lim (n→∞) n²/n³ = 0 ⇒ n² = o(n³)
```

## Master Theorem for Divide-and-Conquer Recurrences

The Master Theorem provides a cookbook solution for recurrences of the form:
```
T(n) = a·T(n/b) + f(n)
where a ≥ 1, b > 1, and f(n) is asymptotically positive
```

### Case 1: Leaf Dominated
If f(n) = O(n^(log_b(a - ε))) for some ε > 0, then:
```
T(n) = Θ(n^(log_b a))
```

### Case 2: Balanced
If f(n) = Θ(n^(log_b a)), then:
```
T(n) = Θ(n^(log_b a) · log n)
```

### Case 3: Root Dominated
If f(n) = Ω(n^(log_b(a + ε))) for some ε > 0, 
and a·f(n/b) ≤ c·f(n) for some c < 1 and large n, then:
```
T(n) = Θ(f(n))
```

### Examples:

1. **Binary Search**:
```
T(n) = T(n/2) + Θ(1)
a=1, b=2 → log_b a = 0
f(n) = Θ(1) = Θ(n^0) → Case 2
Thus, T(n) = Θ(log n)
```

2. **Merge Sort**:
```
T(n) = 2T(n/2) + Θ(n)
a=2, b=2 → log_b a = 1
f(n) = Θ(n) = Θ(n^1) → Case 2
Thus, T(n) = Θ(n log n)
```

3. **Recursive Tree Traversal**:
```
T(n) = 3T(n/4) + Θ(n²)
a=3, b=4 → log_4 3 ≈ 0.793
f(n) = Θ(n²) → Case 3 (n² vs n^0.793)
Check regularity: 3(n/4)² ≤ cn² for c=3/4 < 1
Thus, T(n) = Θ(n²)
```

## Advanced Asymptotic Concepts

### 1. Polynomial vs Exponential
- Polynomial: n^O(1)
- Exponential: 2^O(n)

### 2. Polylogarithmic
- (log n)^O(1)

### 3. Sublinear
- o(n) (e.g., √n, log n)

### 4. Subexponential
- 2^o(n)

## Solving Recurrence Relations

### 1. Substitution Method

**Steps**:

1. Guess the form of solution
2. Use induction to verify
3. Solve for constants

**Example**:
```
T(n) = 2T(n/2) + n
Guess T(n) = O(n log n)
Assume T(k) ≤ ck log k for k < n
Show T(n) ≤ cn log n
```

### 2. Recursion Tree Method

**Steps**:

1. Draw tree of recursive calls
2. Sum costs at each level
3. Sum all levels

**Example**:
```
T(n) = 3T(n/4) + Θ(n²)
Level 0: cn²
Level 1: 3c(n/4)²
Level 2: 9c(n/16)²
...
Sum converges to geometric series
```

### 3. Akra-Bazzi Method (Generalized Master Theorem)

For recurrences of form:
```
T(n) = Σ a_i T(n/b_i) + f(n)
```

Solution:
```
T(n) = Θ(n^p (1 + ∫(f(u)/u^{p+1} du)))
where p satisfies Σ a_i / b_i^p = 1
```

## Common Complexity Classes

| Notation       | Name                  | Example Algorithms               |
|---------------|-----------------------|----------------------------------|
| O(1)          | Constant              | Array access, Hash table lookup  |
| O(log n)      | Logarithmic           | Binary search                    |
| O(n)          | Linear                | Linear search                    |
| O(n log n)    | Linearithmic          | Merge sort, Heap sort            |
| O(n²)         | Quadratic             | Bubble sort, Insertion sort      |
| O(n³)         | Cubic                 | Naive matrix multiplication      |
| O(2^n)        | Exponential           | Subset generation                |
| O(n!)         | Factorial             | Permutation generation           |

## Practical Calculation Examples

### Example 1: Nested Loops
```python
for i in range(n):         # O(n)
    for j in range(i, n):  # O(n-i)
        print(i, j)
```
**Analysis**:
Total operations = Σ (from i=0 to n-1) (n-i) = n + (n-1) + ... + 1 = n(n+1)/2
Thus, O(n²)

### Example 2: Logarithmic Complexity
```python
i = 1
while i < n:     # Log steps
    i *= 2       # Double each time
```
**Analysis**:
Loop runs until 2^k ≥ n ⇒ k = log₂n
Thus, O(log n)

### Example 3: Multiple Terms
```python
for i in range(n):     # O(n)
    for j in range(10): # O(1)
        print(i, j)
for k in range(n):     # O(n)
    print(k)
```
**Analysis**:
First part: O(n) * O(1) = O(n)
Second part: O(n)
Total: O(n) + O(n) = O(n)

## Limitations of Asymptotic Analysis

1. **Hidden constants**: O(n) may be worse than O(n²) for small n
2. **Input characteristics**: Assumes worst-case input
3. **Machine factors**: Ignores hardware-specific optimizations
4. **Recursive overhead**: Function call costs may be significant

## Advanced Topics

### 1. Smoothness Rule
If f(n) is smooth (eventually non-decreasing) and b ≥ 2 is integer, then:
```
f(n) = Θ(f(bn))
```

### 2. Harmonic Series
```
H_n = Σ (1/k) for k=1 to n = Θ(log n)
```

### 3. Stirling's Approximation
```
log(n!) = Θ(n log n)
```

Asymptotic analysis provides the fundamental language for comparing algorithm efficiency, enabling computer scientists to make meaningful comparisons independent of implementation details or hardware considerations.