# Python Methods - Static, Class, and Abstract

Python provides three main types of methods that offer different functionalities and use cases: **Static Methods**, **Class Methods**, and **Abstract Methods**. Understanding these methods is crucial for object-oriented programming and clean code design.

---

## Static Method

A **static method** is a method that belongs to a class but does not require access to the class instance (`self`) or class itself (`cls`). It is defined using the `@staticmethod` decorator.

### Characteristics
- Does not access or modify instance or class-level attributes.
- Useful for utility or helper functions related to the class.

### Syntax

```python
class MyClass:
    @staticmethod
    def static_method(arg1, arg2):
        # Method logic
        return arg1 + arg2
```

### Example

```python
class Calculator:
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def subtract(a, b):
        return a - b

# Using static methods
print(Calculator.add(5, 3))       # Output: 8
print(Calculator.subtract(10, 4)) # Output: 6
```

---

## Class Method

A **class method** is a method that operates on the class level rather than the instance level. It is defined using the `@classmethod` decorator and takes `cls` (class itself) as its first parameter.

### Characteristics
- Can modify class-level attributes.
- Often used as factory methods to create instances.

### Syntax

```python
class MyClass:
    @classmethod
    def class_method(cls, arg1):
        # Method logic
        return cls(arg1)
```

### Example

```python
class Employee:
    company = "TechCorp"

    def __init__(self, name):
        self.name = name

    @classmethod
    def change_company(cls, new_company):
        cls.company = new_company

# Using class methods
print(Employee.company)  # Output: TechCorp
Employee.change_company("InnovateInc")
print(Employee.company)  # Output: InnovateInc
```

---

## Abstract Method

An **abstract method** is a method that is declared but not implemented. It serves as a blueprint for derived classes. Abstract methods are defined in abstract base classes (ABCs), which require subclasses to implement the abstract methods.

### Characteristics
- Defined using the `@abstractmethod` decorator from the `abc` module.
- Enforces implementation in subclasses.
- Abstract classes cannot be instantiated directly.

### Syntax

```python
from abc import ABC, abstractmethod

class MyAbstractClass(ABC):
    @abstractmethod
    def abstract_method(self):
        pass
```

### Example

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"

# Instantiating subclasses
dog = Dog()
cat = Cat()
print(dog.make_sound())  # Output: Woof!
print(cat.make_sound())  # Output: Meow!
```

---

## Comparison Table

| Feature               | Static Method                 | Class Method               | Abstract Method                |
|-----------------------|-------------------------------|----------------------------|---------------------------------|
| **Decorator**         | `@staticmethod`              | `@classmethod`             | `@abstractmethod`              |
| **First Argument**    | None                         | `cls`                      | Must be overridden by subclass |
| **Access Class Data** | No                           | Yes                        | Depends on subclass           |
| **Access Instance Data** | No                        | No                         | Depends on subclass           |
| **Use Case**          | Utility functions            | Factory methods, class-wide settings | Enforce subclass behavior |

---

## Key Takeaways

- **Static Methods**: Independent of instance and class. Great for utility functions.
- **Class Methods**: Operate at the class level. Often used for class-level changes or factory methods.
- **Abstract Methods**: Define a contract for subclasses, ensuring consistent implementation.

Understanding these methods helps you write clean, maintainable, and modular code. They each serve unique purposes and should be used appropriately in different scenarios.
