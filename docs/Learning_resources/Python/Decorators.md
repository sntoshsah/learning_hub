# Python First-Class Functions, Closures, and Decorators

## Overview

This guide covers three fundamental concepts in Python: **First-Class Functions**, **Closures**, and **Decorators**. These concepts form the backbone of functional programming in Python and are essential for creating dynamic, reusable, and modular code.

---

## First-Class Functions

### What Are First-Class Functions?

In Python, functions are treated as first-class citizens, which means:

- They can be assigned to variables.
- They can be passed as arguments to other functions.
- They can be returned from other functions.
- They can be stored in data structures such as lists or dictionaries.

### Syntax and Examples

```python
# Assigning a function to a variable
def greet(name):
    return f"Hello, {name}!"

say_hello = greet
print(say_hello("Alice"))  # Output: Hello, Alice!

# Passing a function as an argument

def execute_function(func, value):
    return func(value)

print(execute_function(greet, "Bob"))  # Output: Hello, Bob!

# Returning a function from another function

def outer_function():
    def inner_function():
        return "Hello from inner function!"
    return inner_function

inner = outer_function()
print(inner())  # Output: Hello from inner function!
```

---

## Closures

### What Are Closures?

A **closure** is a function that retains access to the variables from its enclosing scope even after that scope has finished executing. Closures are created when:

1. A nested function references variables from the outer function.
2. The outer function returns the nested function.

### Syntax and Examples

```python
# Example of a closure

def outer_function(message):
    def inner_function():
        return f"Message: {message}"
    return inner_function

closure_func = outer_function("Hello, World!")
print(closure_func())  # Output: Message: Hello, World!

# The 'message' variable is retained even after outer_function finishes.
```

### Real-World Use Case

Closures are often used to create function factories or to maintain state between function calls.

```python
# Function factory example

def multiplier(factor):
    def multiply_by_factor(number):
        return number * factor
    return multiply_by_factor

multiply_by_2 = multiplier(2)
multiply_by_3 = multiplier(3)

print(multiply_by_2(10))  # Output: 20
print(multiply_by_3(10))  # Output: 30
```

---

## Decorators

### What Are Decorators?

A **decorator** is a higher-order function that modifies or extends the behavior of another function or method without modifying its source code. Decorators are commonly used for:

- Logging
- Access control
- Caching
- Measuring execution time

### Syntax and Examples

Decorators are typically implemented using the `@decorator_name` syntax.

```python
# Basic decorator example

def decorator(func):
    def wrapper():
        print("Before the function call")
        func()
        print("After the function call")
    return wrapper

@decorator
def say_hello():
    print("Hello, Decorators!")

say_hello()
# Output:
# Before the function call
# Hello, Decorators!
# After the function call
```

### Decorating Functions With Arguments

```python
# Decorator for functions with arguments

def decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Arguments: {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result
    return wrapper

@decorator
def add(a, b):
    return a + b

add(2, 3)
# Output:
# Arguments: (2, 3), {}
# Result: 5
```

### Built-in Decorators

Python provides several built-in decorators, such as:

- `@staticmethod`: Defines a static method.
- `@classmethod`: Defines a class method.
- `@property`: Defines a property method.

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

circle = Circle(5)
print(circle.radius)  # Output: 5
circle.radius = 10
print(circle.radius)  # Output: 10
```

---

## Summary

| Concept         | Description                                                                                         | Example Use Case                            |
|-----------------|-----------------------------------------------------------------------------------------------------|---------------------------------------------|
| **First-Class Functions** | Treat functions as values: assign, pass, or return them.                                           | Callbacks, function factories               |
| **Closures**    | Retain access to variables in an enclosing scope after the scope has exited.                         | Function factories, maintaining state       |
| **Decorators**  | Modify or extend the behavior of functions or methods without altering their source code.            | Logging, access control, performance timing |

These concepts enable Python developers to write cleaner, more modular, and more reusable code.
