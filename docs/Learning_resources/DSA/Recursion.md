
# Recursion in Computer Science

## Principle of Recursion

Recursion is a programming technique where a **function calls itself** to solve a problem by breaking it down into smaller subproblems. 

### Key Components:
1. **Base Case**: The simplest instance of the problem that can be solved directly (stopping condition)
2. **Recursive Case**: The part where the function calls itself with a modified input to make progress toward the base case

### How Recursion Works:
- Each recursive call creates a new instance of the function on the call stack
- The stack unwinds when base cases are reached
- Results propagate back up through the call chain

```python
def countdown(n):
    """Simple recursive countdown function"""
    if n <= 0:  # Base case
        print("Blastoff!")
    else:
        print(n)
        countdown(n - 1)  # Recursive call

countdown(5)
# Output:
# 5
# 4
# 3
# 2
# 1
# Blastoff!
```

## Comparison Between Recursion and Iteration

| Characteristic | Recursion | Iteration |
|---------------|----------|-----------|
| **Definition** | Function calls itself | Loops (for, while) repeat code |
| **Termination** | Base case stops recursion | Condition stops loop |
| **State** | Maintained on call stack | Explicitly maintained in variables |
| **Memory** | Higher (stack frames) | Lower (fixed variables) |
| **Code Size** | Often smaller | Often larger |
| **Readability** | Better for recursive problems | Better for simple repetition |
| **Speed** | Slower (function call overhead) | Faster |

```python
# Iterative vs Recursive Factorial
def factorial_iterative(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result

def factorial_recursive(n):
    if n == 0 or n == 1:  # Base case
        return 1
    return n * factorial_recursive(n - 1)  # Recursive case

print(factorial_iterative(5))  # 120
print(factorial_recursive(5))  # 120
```

## Tail Recursion

Tail recursion occurs when the **recursive call is the last operation** in the function. Some languages can optimize this to avoid stack growth.

### Characteristics:
- No computation after the recursive call
- Can be optimized to use constant stack space
- Python doesn't optimize tail recursion (unlike some functional languages)

```python
# Regular vs Tail Recursive Factorial
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)  # Not tail recursive (multiplication after call)

def factorial_tail(n, accumulator=1):
    if n == 0:
        return accumulator
    return factorial_tail(n - 1, n * accumulator)  # Tail recursive

print(factorial(5))         # 120
print(factorial_tail(5))    # 120
```

## Classic Recursive Problems

### Factorial
```python
def factorial(n):
    """Computes n! recursively"""
    if n == 0 or n == 1:  # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case

print(factorial(5))  # 120
```

### Fibonacci Sequence
```python
def fibonacci(n):
    """Returns nth Fibonacci number"""
    if n <= 1:  # Base cases
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)  # Recursive case

# More efficient with memoization
def fib_memo(n, memo={0:0, 1:1}):
    if n not in memo:
        memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

print(fibonacci(10))  # 55 (but inefficient)
print(fib_memo(100))  # 354224848179261915075 (efficient)
```

### Greatest Common Divisor (GCD)
```python
def gcd(a, b):
    """Euclidean algorithm for GCD"""
    if b == 0:  # Base case
        return a
    return gcd(b, a % b)  # Recursive case

print(gcd(48, 18))  # 6
```

### Tower of Hanoi
```python
def tower_of_hanoi(n, source, target, auxiliary):
    """Solve Tower of Hanoi problem"""
    if n == 1:  # Base case
        print(f"Move disk 1 from {source} to {target}")
        return
    # Move n-1 disks from source to auxiliary
    tower_of_hanoi(n-1, source, auxiliary, target)
    # Move remaining disk from source to target
    print(f"Move disk {n} from {source} to {target}")
    # Move n-1 disks from auxiliary to target
    tower_of_hanoi(n-1, auxiliary, target, source)

tower_of_hanoi(3, 'A', 'C', 'B')
# Output:
# Move disk 1 from A to C
# Move disk 2 from A to B
# Move disk 1 from C to B
# Move disk 3 from A to C
# Move disk 1 from B to A
# Move disk 2 from B to C
# Move disk 1 from A to C
```

## Applications of Recursion

### Common Use Cases:
1. **Tree/Graph Traversals** (DFS, tree operations)
2. **Divide and Conquer Algorithms** (merge sort, quick sort)
3. **Backtracking Problems** (maze solving, N-queens)
4. **Mathematical Computations** (fractals, combinatorics)
5. **Parsing and Syntax Analysis** (compiler design)

```python
# Binary Tree Traversal
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def inorder_traversal(node):
    if node:
        inorder_traversal(node.left)
        print(node.value, end=" ")
        inorder_traversal(node.right)

# Create a simple tree
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

print("Inorder traversal:")
inorder_traversal(root)  # Output: 4 2 5 1 3

# Directory Traversal
import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

# list_files('/path/to/directory')  # Uncomment to use
```

## Efficiency of Recursion

### Advantages:
1. **Elegant solutions** for inherently recursive problems
2. **Reduces code complexity** for certain problems
3. **Natural fit** for tree/graph structures and mathematical definitions

### Disadvantages:
1. **Stack overflow risk** with deep recursion
2. **Memory overhead** from stack frames
3. **Slower execution** due to function call overhead
4. **Debugging complexity** compared to iteration

### Optimization Techniques:
1. **Memoization**: Cache results of expensive function calls
2. **Tail Recursion**: Where supported by language
3. **Iterative Conversion**: Rewrite recursive algorithms iteratively

```python
# Memoization Example
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

print(fib(100))  # 354224848179261915075 (instant with memoization)

# Iterative Fibonacci (more efficient)
def fib_iter(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(fib_iter(100))  # Same result, faster execution
```

### When to Use Recursion:
- Problem has natural recursive structure (trees, divide-and-conquer)
- Recursive solution is significantly simpler
- Depth is limited and stack overflow isn't a concern
- Language supports tail call optimization

### When to Avoid Recursion:
- Performance is critical
- Problem depth could lead to stack overflow
- Iterative solution is straightforward
- Working with very large data sets