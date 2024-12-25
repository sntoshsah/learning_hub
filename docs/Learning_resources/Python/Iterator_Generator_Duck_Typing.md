# Python Generators, Iterators, and Duck Typing

## Overview

This page explores three foundational concepts in Python: **Generators**, **Iterators**, and **Duck Typing**. These concepts play a significant role in Python's design philosophy and practical programming.

---

## Python Generators

### What are Generators?
Generators are a type of iterable that yield values one at a time, allowing you to iterate over data without storing the entire dataset in memory. They are defined using the `yield` keyword instead of `return`.

### Characteristics
- Generators are memory-efficient.
- They produce items only when needed (lazy evaluation).
- They maintain their state between iterations.

### Syntax

```python
def generator_function():
    yield 1
    yield 2
    yield 3

# Using the generator
gen = generator_function()
print(next(gen))  # Output: 1
print(next(gen))  # Output: 2
print(next(gen))  # Output: 3
```

### Example: Generating Fibonacci Sequence

```python
def fibonacci(limit):
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b

# Using the generator
for num in fibonacci(10):
    print(num)  # Output: 0 1 1 2 3 5 8
```

---

## Python Iterators

### What are Iterators?
An iterator is an object that implements two methods:
- `__iter__()` - Returns the iterator object itself.
- `__next__()` - Returns the next item from the iterator. If no items remain, it raises `StopIteration`.

### Characteristics
- Used to traverse through all items in a collection (e.g., list, tuple, set).
- Iterators are consumed once.

### Syntax

```python
class MyIterator:
    def __init__(self, numbers):
        self.numbers = numbers
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.numbers):
            value = self.numbers[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration

# Using the iterator
nums = MyIterator([1, 2, 3])
for num in nums:
    print(num)  # Output: 1 2 3
```

### Iterables vs Iterators

| Feature         | Iterable                       | Iterator                        |
|-----------------|--------------------------------|---------------------------------|
| Definition      | An object that supports `__iter__()` | An object that supports `__iter__()` and `__next__()` |
| Example         | Lists, tuples, dictionaries    | Objects like file readers, generators |
| Consumption     | Can be converted into an iterator | Consumed item by item         |

---

## Duck Typing

### What is Duck Typing?
Duck typing is a programming concept where the suitability of an object is determined by the presence of certain methods and properties, rather than the object's type. This aligns with Python's dynamic typing philosophy.

> "If it looks like a duck, swims like a duck, and quacks like a duck, then it probably is a duck."

### Characteristics
- Focuses on behavior rather than inheritance or explicit type checking.
- Encourages flexibility and adaptability in code.

### Syntax

```python
class Duck:
    def quack(self):
        print("Quack!")

    def swim(self):
        print("Swimming like a duck.")

class Person:
    def quack(self):
        print("I can quack too!")

    def swim(self):
        print("I can swim as well!")

# Function demonstrating Duck Typing
def interact_with_duck(duck):
    duck.quack()
    duck.swim()

# Using Duck Typing
interact_with_duck(Duck())  # Output: Quack! Swimming like a duck.
interact_with_duck(Person())  # Output: I can quack too! I can swim as well!
```

### Advantages
- Encourages writing more generic and reusable code.
- Simplifies code without requiring strict type hierarchies.

### Limitations
- May lead to runtime errors if the expected behavior is not implemented in the object.

---

## Summary Table

| Concept     | Description                              | Key Methods              | Example Use Cases                          |
|-------------|------------------------------------------|--------------------------|-------------------------------------------|
| Generators  | Yield values lazily                     | `yield`                  | Large datasets, streams                   |
| Iterators   | Object for sequential data traversal    | `__iter__`, `__next__`   | File handling, custom collections         |
| Duck Typing | Behavior-based object suitability       | N/A                      | Polymorphism without inheritance          |

---

Let me know if there is anything else you would like to explore in more detail!
