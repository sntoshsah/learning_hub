# Dynamic Programming: Concepts and Applications

## Introduction to Dynamic Programming

Dynamic Programming (DP) is a powerful algorithmic technique for solving complex problems by breaking them down into simpler subproblems. It follows the principle of **optimal substructure** (problems can be solved optimally by combining optimal solutions to subproblems) and **overlapping subproblems** (the same subproblems are solved multiple times).

### Key Characteristics:
1. **Memoization**: Storing results of expensive function calls
2. **Tabulation**: Building a table bottom-up to store solutions
3. **Reuse**: Avoiding recomputation by storing intermediate results

## When to Use Dynamic Programming

DP is effective for problems that exhibit:
- Optimal substructure
- Overlapping subproblems
- Recursive nature
- Combinatorial complexity

## Classic DP Problems with Python Implementations

### 1. Fibonacci Sequence

**Problem**: Compute the nth Fibonacci number efficiently.

#### Naive Recursive Approach (O(2^n))
```python
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
```

#### DP Solution with Memoization (O(n))
```python
def fib_memo(n, memo={0:0, 1:1}):
    if n not in memo:
        memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]
```

#### DP Solution with Tabulation (O(n))
```python
def fib_tab(n):
    if n <= 1:
        return n
    dp = [0] * (n+1)
    dp[1] = 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

### 2. 0/1 Knapsack Problem

**Problem**: Given weights and values of items, put them in a knapsack of capacity W to get maximum value.

#### DP Solution (O(nW))
```python
def knapsack(W, wt, val, n):
    dp = [[0 for _ in range(W+1)] for _ in range(n+1)]
    
    for i in range(1, n+1):
        for w in range(1, W+1):
            if wt[i-1] <= w:
                dp[i][w] = max(val[i-1] + dp[i-1][w-wt[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][W]

# Example
val = [60, 100, 120]
wt = [10, 20, 30]
W = 50
n = len(val)
print(knapsack(W, wt, val, n))  # Output: 220
```

### 3. Longest Common Subsequence (LCS)

**Problem**: Find the length of the longest subsequence present in both strings.

#### DP Solution (O(mn))
```python
def lcs(X, Y):
    m, n = len(X), len(Y)
    dp = [[0]*(n+1) for _ in range(m+1)]
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# Example
X = "AGGTAB"
Y = "GXTXAYB"
print(lcs(X, Y))  # Output: 4 ("GTAB")
```

### 4. Coin Change Problem

**Problem**: Find the minimum number of coins needed to make change for a given amount.

#### DP Solution (O(amount * num_coins))
```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

# Example
coins = [1, 2, 5]
amount = 11
print(coin_change(coins, amount))  # Output: 3 (5 + 5 + 1)
```

### 5. Matrix Chain Multiplication

**Problem**: Find the optimal way to multiply matrices to minimize operations.

#### DP Solution (O(n^3))
```python
def matrix_chain_order(p):
    n = len(p) - 1
    m = [[0 for _ in range(n)] for _ in range(n)]
    s = [[0 for _ in range(n)] for _ in range(n)]
    
    for l in range(2, n+1):
        for i in range(n-l+1):
            j = i + l - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                cost = m[i][k] + m[k+1][j] + p[i]*p[k+1]*p[j+1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    return m[0][n-1]

# Example
arr = [1, 2, 3, 4]
print(matrix_chain_order(arr))  # Output: 18
```

### 6. Longest Increasing Subsequence (LIS)

**Problem**: Find the length of the longest subsequence that is strictly increasing.

#### DP Solution (O(n^2))
```python
def lis(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j]+1)
    return max(dp)

# Example
nums = [10, 9, 2, 5, 3, 7, 101, 18]
print(lis(nums))  # Output: 4 (2, 3, 7, 101)
```

### 7. Edit Distance

**Problem**: Find the minimum operations (insert, delete, replace) to convert one string to another.

#### DP Solution (O(mn))
```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
        
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # Delete
                                   dp[i][j-1],    # Insert
                                   dp[i-1][j-1])  # Replace
    return dp[m][n]

# Example
word1 = "horse"
word2 = "ros"
print(min_distance(word1, word2))  # Output: 3
```


**DP Problem-Solving Approach:**
Step-by-Step Methodology:

1. Define the Subproblem: Identify how to break the problem down

2. Formulate the Recurrence: Express solution in terms of smaller subproblems

3. Identify Base Cases: Define the simplest subproblems

4. Choose Implementation:

    Top-down with memoization (recursive)

    Bottom-up with tabulation (iterative)

Compute the Solution: Build up the solution from subproblems

**Implementation Patterns:**
1. Top-Down with Memoization
```python

def dp_top_down(params, memo={}):
    if params in memo:
        return memo[params]
    # Base cases
    if base_case_condition:
        return base_case_value
    # Recursive case
    result = compute_result_using_subproblems
    memo[params] = result
    return result
```
2. Bottom-Up with Tabulation
```python

def dp_bottom_up(params):
    # Initialize DP table
    dp = initialize_table
    
    # Base cases
    dp[base_case] = base_case_value
    
    # Fill table iteratively
    for subproblem in all_subproblems:
        dp[subproblem] = compute_from_smaller_subproblems
    
    return dp[final_problem]
```

### Advanced DP Problems
1. Matrix Chain Multiplication

Problem Statement: Given a sequence of matrices, find the most efficient way to multiply them (parenthesization) to minimize operations.

DP Solution (O(n³)):
```python

def matrix_chain_order(p):
    n = len(p) - 1
    m = [[0 for _ in range(n)] for _ in range(n)]
    s = [[0 for _ in range(n)] for _ in range(n)]
    
    for l in range(2, n+1):  # l is chain length
        for i in range(n - l + 1):
            j = i + l - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                cost = m[i][k] + m[k+1][j] + p[i]*p[k+1]*p[j+1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    
    return m, s

def print_optimal_parens(s, i, j):
    if i == j:
        print(f"A{i+1}", end="")
    else:
        print("(", end="")
        print_optimal_parens(s, i, s[i][j])
        print_optimal_parens(s, s[i][j]+1, j)
        print(")", end="")

# Example
p = [30, 35, 15, 5, 10, 20, 25]  # Dimensions of matrices
m, s = matrix_chain_order(p)
print("Minimum multiplications:", m[0][len(p)-2])
print("Optimal parenthesization: ", end="")
print_optimal_parens(s, 0, len(p)-2)  # Output: ((A1(A2A3))((A4A5)A6))
```
2. Longest Increasing Subsequence (LIS)

Problem Statement: Find the length of the longest subsequence of a given sequence such that all elements are sorted in increasing order.

DP Solution (O(n²)):
```python

def lis(arr):
    n = len(arr)
    dp = [1]*n
    
    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j] and dp[i] < dp[j] + 1:
                dp[i] = dp[j] + 1
    
    return max(dp)

# Optimized O(n log n) solution
import bisect

def lis_optimized(arr):
    tails = []
    for num in arr:
        idx = bisect.bisect_left(tails, num)
        if idx == len(tails):
            tails.append(num)
        else:
            tails[idx] = num
    return len(tails)

# Example
arr = [10, 22, 9, 33, 21, 50, 41, 60]
print("LIS length (DP):", lis(arr))  # 5
print("LIS length (Optimized):", lis_optimized(arr))  # 5
```
**DP Optimization Techniques**

- Space Optimization: Reduce space complexity by reusing DP table rows

- State Compression: Represent DP states more compactly

- Knuth Optimization: For certain DP problems with quadrangle inequalities

- Convex Hull Trick: For DP problems with specific recurrence relations

Space Optimized 0/1 Knapsack Example:
```python

def knapsack_space_optimized(W, wt, val, n):
    dp = [0]*(W+1)
    
    for i in range(1, n+1):
        for w in range(W, 0, -1):  # Reverse order to prevent overwriting
            if wt[i-1] <= w:
                dp[w] = max(dp[w], dp[w-wt[i-1]] + val[i-1])
    
    return dp[W]
```
**When to Use Dynamic Programming**

- The problem can be broken down into overlapping subproblems
- The problem has optimal substructure property
- The problem requires optimization (min/max) or counting
- The naive recursive solution has exponential time complexity


## DP Problem-Solving Approach

1. **Identify the subproblems**: What are the smaller versions of the problem?
2. **Define the recurrence relation**: How do solutions to subproblems combine?
3. **Implement memoization or tabulation**: Choose top-down or bottom-up
4. **Handle base cases**: What are the simplest cases?
5. **Compute the final solution**: Combine subproblem solutions

## Advanced DP Concepts

### 1. State Compression
Reduce space complexity by reusing DP arrays.

```python
# Fibonacci with O(1) space
def fib_space_optimized(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b
```

### 2. Bitmask DP
For problems involving subsets or permutations.

```python
# Traveling Salesman Problem (TSP)
def tsp(dist):
    n = len(dist)
    memo = {}
    
    def dp(mask, pos):
        if mask == (1 << n) - 1:
            return dist[pos][0]
        if (mask, pos) in memo:
            return memo[(mask, pos)]
        
        ans = float('inf')
        for city in range(n):
            if not (mask & (1 << city)):
                ans = min(ans, dist[pos][city] + dp(mask | (1 << city), city))
        memo[(mask, pos)] = ans
        return ans
    
    return dp(1, 0)

# Example
dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
print(tsp(dist))  # Output: 80
```

## Common DP Patterns

1. **Prefix/Suffix DP**: LIS, LCS
2. **Interval DP**: Matrix chain multiplication
3. **Tree DP**: Problems on trees
4. **Digit DP**: Counting problems with digit constraints
5. **Probability DP**: Problems involving probabilities

Dynamic programming is a versatile technique that, when mastered, can solve a wide range of complex problems efficiently. The key is to recognize the optimal substructure and overlapping subproblems in the problem at hand.