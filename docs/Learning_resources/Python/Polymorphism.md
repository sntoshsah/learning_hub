# Polymorphism in Python

Polymorphism is a fundamental concept in Object-Oriented Programming (OOP). It allows objects of different classes to be treated as objects of a common superclass. The word "polymorphism" means "many forms," and it is implemented in Python through method overriding, method overloading, and interfaces.

This document provides a detailed explanation of polymorphism in Python with examples and syntax.

---

## Types of Polymorphism in Python

### 1. **Method Overriding**
Method overriding occurs when a subclass provides a specific implementation of a method already defined in its superclass. The subclass's method is called instead of the parent class's method when invoked through the subclass instance.

#### Syntax:
```python
class ParentClass:
    def method_name(self):
        # Parent method implementation

class ChildClass(ParentClass):
    def method_name(self):
        # Overridden method implementation
```

#### Example:
```python
class Animal:
    def sound(self):
        return "Generic animal sound"

class Dog(Animal):
    def sound(self):
        return "Woof"

class Cat(Animal):
    def sound(self):
        return "Meow"

# Polymorphic behavior
animals = [Dog(), Cat(), Animal()]
for animal in animals:
    print(animal.sound())
```
**Output:**
```
Woof
Meow
Generic animal sound
```

---

### 2. **Method Overloading (Simulated)**
Python does not natively support method overloading as other languages do, but similar behavior can be achieved by defining a method that accepts varying numbers of arguments using default parameters or `*args`.

#### Syntax:
```python
class ClassName:
    def method_name(self, *args):
        # Implementation depending on the arguments
```

#### Example:
```python
class Calculator:
    def add(self, *args):
        return sum(args)

calc = Calculator()
print(calc.add(1, 2))          # Output: 3
print(calc.add(1, 2, 3, 4))   # Output: 10
```

---

### 3. **Polymorphism with Functions and Objects**
Polymorphism allows the same function to operate on different types of objects. This is achieved by writing a function that accepts objects of different classes, as long as those objects implement the required methods.

#### Example:
```python
class Bird:
    def fly(self):
        return "Birds can fly"

class Penguin:
    def fly(self):
        return "Penguins cannot fly"

def flying_ability(bird):
    print(bird.fly())

sparrow = Bird()
pingu = Penguin()

flying_ability(sparrow)  # Output: Birds can fly
flying_ability(pingu)    # Output: Penguins cannot fly
```

---

### 4. **Polymorphism with Abstract Base Classes (ABCs)**
Abstract base classes in Python can enforce the implementation of methods in derived classes, providing a structured way to implement polymorphism.

#### Syntax:
```python
from abc import ABC, abstractmethod

class AbstractClass(ABC):
    @abstractmethod
    def abstract_method(self):
        pass

class ConcreteClass(AbstractClass):
    def abstract_method(self):
        # Implementation
```

#### Example:
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius * self.radius

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

shapes = [Circle(5), Rectangle(4, 6)]
for shape in shapes:
    print(f"Area: {shape.area()}")
```
**Output:**
```
Area: 78.5
Area: 24
```

---

## Benefits of Polymorphism

1. **Flexibility:** A single interface can work with different data types or objects.
2. **Code Reusability:** Reduces duplication by allowing methods to work with objects of different classes.
3. **Scalability:** Makes programs easier to extend as new object types can be added with minimal changes to existing code.

---

Polymorphism is a powerful concept that makes object-oriented programming in Python both flexible and maintainable. By using method overriding, simulated method overloading, or abstract base classes, developers can build robust applications that are easy to scale and maintain.
