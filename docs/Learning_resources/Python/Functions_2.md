# Python Functions-2 - A Comprehensive Guide

Functions are reusable blocks of code designed to perform specific tasks. Python supports various types of functions, including built-in, user-defined, and anonymous (lambda) functions. This guide covers detailed explanations, syntax, and examples for various types of Python functions.

---

## 1. Lambda Functions

**Lambda functions** are anonymous, single-expression functions often used for short, concise operations.

### Syntax
```python
lambda arguments: expression
```

### Example
```python
# Add two numbers
add = lambda x, y: x + y
print(add(5, 3))  # Output: 8

# Square of a number
square = lambda x: x ** 2
print(square(4))  # Output: 16
```

---

## 2. Recursive Functions

**Recursive functions** are functions that call themselves to solve smaller instances of a problem.

### Syntax
```python
def function_name(parameters):
    if termination_condition:
        return base_case
    else:
        return function_name(smaller_problem)
```

### Example
```python
# Factorial of a number
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # Output: 120
```

---

## 3. Map Functions

**Map functions** apply a given function to each item in an iterable (e.g., list, tuple).

### Syntax
```python
map(function, iterable)
```

### Example
```python
numbers = [1, 2, 3, 4]
squares = map(lambda x: x ** 2, numbers)
print(list(squares))  # Output: [1, 4, 9, 16]
```

---

## 4. Filter Functions

**Filter functions** filter elements from an iterable based on a condition.

### Syntax
```python
filter(function, iterable)
```

### Example
```python
numbers = [1, 2, 3, 4, 5]
even_numbers = filter(lambda x: x % 2 == 0, numbers)
print(list(even_numbers))  # Output: [2, 4]
```

---

## 5. Reduce Functions

**Reduce functions**, from the `functools` module, reduce an iterable to a single cumulative value.

### Syntax
```python
from functools import reduce
reduce(function, iterable)
```

### Example
```python
from functools import reduce
numbers = [1, 2, 3, 4]
product = reduce(lambda x, y: x * y, numbers)
print(product)  # Output: 24
```

---

## 6. Functools Module

The `functools` module provides higher-order functions for functional programming.

### Functions in `functools`
1. `cmp_to_key(func)`: Converts a comparison function into a key function.
2. `lru_cache(maxsize=None)`: Decorator to cache results for optimization.
3. `partial(func, *args, **keywords)`: Create a new function with partial application of arguments.
4. `reduce(func, iterable)`: Applies a function cumulatively to items in an iterable.
5. `total_ordering(cls)`: Class decorator to fill in missing ordering methods.
6. `wraps(wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES)`: Decorator to update a wrapper function.

### Example: Partial Function
```python
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
print(square(5))  # Output: 25
```

---

## 7. Currying Functions

**Currying** transforms a function with multiple arguments into a sequence of functions, each with a single argument.

### Example
```python
def curry(func):
    def curried(x):
        return lambda y: func(x, y)
    return curried

def add(x, y):
    return x + y

curried_add = curry(add)
add_five = curried_add(5)
print(add_five(10))  # Output: 15
```

---

## 8. Memoization Functions

**Memoization** is a technique to store function results for faster future computation.

### Example
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))  # Output: 55
```

---

## 9. Threading Functions

**Threading** allows parallel execution of code.

### Syntax
```python
from threading import Thread
```

### Example
```python
from threading import Thread

def print_numbers():
    for i in range(5):
        print(i)

def print_letters():
    for letter in 'abcde':
        print(letter)

# Create threads
thread1 = Thread(target=print_numbers)
thread2 = Thread(target=print_letters)

# Start threads
thread1.start()
thread2.start()

# Wait for threads to complete
thread1.join()
thread2.join()
```

---

## Additional Functions

### Higher-Order Functions

Higher-order functions take other functions as arguments or return them.

**Example**
```python
def apply_function(func, value):
    return func(value)

def square(x):
    return x ** 2

print(apply_function(square, 5))  # Output: 25
```

---

This guide provides a solid foundation for understanding and using Python functions effectively. Feel free to expand further based on your use case!
