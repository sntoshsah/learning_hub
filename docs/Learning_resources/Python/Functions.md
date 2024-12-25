# Python Functions - A Comprehensive Guide

## Overview

A **function** in Python is a block of reusable code that performs a specific task. Functions help to modularize the program, making it easier to write, debug, and maintain.

### Key Features
- **Modularization**: Breaks down the code into manageable chunks.
- **Reusability**: Code can be reused in different parts of the program.
- **Parameterization**: Accepts inputs (parameters) to customize its behavior.
- **Return Values**: Produces an output that can be used elsewhere in the program.

---

## Defining a Function

Functions in Python are defined using the `def` keyword, followed by the function name and parentheses `()` containing optional parameters.

### Syntax
```python
def function_name(parameters):
    """Optional Docstring"""
    # Function body
    return value
```

### Example
```python
def greet(name):
    """Greets the user by name."""
    return f"Hello, {name}!"

# Calling the function
print(greet("Alice"))  # Output: Hello, Alice!
```

---

## Types of Functions

1. **Built-in Functions**: Provided by Python, such as `len()`, `print()`, etc.
2. **User-defined Functions**: Created by the user to perform specific tasks.
3. **Anonymous Functions (Lambdas)**: Functions without a name, defined using the `lambda` keyword.

---

## Function Parameters and Arguments

### Types of Parameters
1. **Positional Parameters**: Passed in order.
2. **Default Parameters**: Provide default values.
3. **Keyword Parameters**: Explicitly specify parameter names.
4. **Variable-length Parameters**: Allow multiple arguments.

### Examples
```python 
def describe_person(name, age=25, *hobbies, **attributes):
    print(f"Name: {name}, Age: {age}")
    print(f"Hobbies: {hobbies}")
    print(f"Attributes: {attributes}")

describe_person("Bob", 30, "reading", "cycling", height=180, weight=75)
```
Output:

```python exec="on"
def describe_person(name, age=25, *hobbies, **attributes):
    print(f"Name: {name}, \t Age: {age}\n")
    print(f"Hobbies: {hobbies}\n")
    print(f"Attributes: {attributes}")

describe_person("Bob", 30, "reading", "cycling", height=180, weight=75)
```
---

## Returning Values

Functions can return values using the `return` keyword. A function without a `return` statement implicitly returns `None`.

```python
def add(a, b):
    return a + b

result = add(5, 3)
print(result)  # Output: 8
```

---

# Built-in Functions

Python provides numerous built-in functions to perform common tasks. Here are some commonly used ones:

| Category       | Function Examples                            |
|----------------|---------------------------------------------|
| **Type**       | `type()`, `isinstance()`, `id()`            |
| **Math**       | `abs()`, `round()`, `min()`, `max()`        |
| **String**     | `len()`, `str()`, `ord()`, `chr()`          |
| **Iterables**  | `len()`, `sum()`, `sorted()`, `zip()`       |
| **Input/Output**| `print()`, `input()`                       |
| **Conversion** | `int()`, `float()`, `str()`, `list()`, `dict()` |

### Examples
```python
# Type functions
x = 10
print(type(x))  # Output: <class 'int'>

# Math functions
print(abs(-5))  # Output: 5
print(round(4.567, 2))  # Output: 4.57

# String functions
print(len("hello"))  # Output: 5
print(ord('A'))  # Output: 65
```

---

# Magic Methods (Dunder Methods)

Magic methods, also known as dunder (double underscore) methods, are special methods that allow you to define custom behaviors for built-in operations. These methods start and end with double underscores, e.g., `__init__`, `__str__`, `__add__`.

### Common Magic Methods

| Method       | Purpose                                       |
|--------------|----------------------------------------------|
| `__init__`   | Constructor (initialization of objects)      |
| `__str__`    | String representation of an object           |
| `__repr__`   | Official string representation of an object  |
| `__len__`    | Defines behavior for `len()`                 |
| `__add__`    | Defines behavior for `+` operator            |
| `__getitem__`| Defines behavior for indexing (`[]`)         |

### Examples
```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point({self.x}, {self.y})"

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

# Using the class
p1 = Point(1, 2)
p2 = Point(3, 4)
print(p1)  # Output: Point(1, 2)
print(p1 + p2)  # Output: Point(4, 6)
```

---

# Lambda Functions

Lambda functions are small, anonymous functions defined with the `lambda` keyword. They can have any number of arguments but only one expression.

### Syntax
```python
lambda arguments: expression
```

### Example
```python
# Regular function
add = lambda x, y: x + y
print(add(5, 3))  # Output: 8

# Using lambda with `map`
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, numbers))
print(squared)  # Output: [1, 4, 9, 16]
```

---

# Key Differences Between Regular and Lambda Functions

| Feature                  | Regular Function                 | Lambda Function                  |
|--------------------------|----------------------------------|-----------------------------------|
| **Definition**           | Uses `def` keyword              | Uses `lambda` keyword            |
| **Name**                 | Named or anonymous              | Anonymous                        |
| **Number of Expressions**| Multiple expressions/statements | Single expression                |
| **Readability**          | More readable                   | Concise but less readable        |

---

This document provides a comprehensive understanding of Python functions, built-in functions, magic methods, and lambda expressions. Feel free to experiment with these concepts to deepen your understanding!
