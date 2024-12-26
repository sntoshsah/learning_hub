# Object-Oriented Programming (OOPs) Concepts

Object-Oriented Programming (OOP) is a programming paradigm based on the concept of "objects," which can contain data in the form of fields (often called attributes) and code in the form of methods. OOP aims to implement real-world entities like inheritance, polymorphism, abstraction, and encapsulation in programming.

This document provides a detailed explanation of OOP concepts with Python examples.

---

## Core OOP Concepts

### 1. **Class**
A class is a blueprint for creating objects. It defines the attributes and methods that the objects created from the class will have.

#### Syntax:
```python
class ClassName:
    # Class attributes
    attribute = value

    # Constructor
    def __init__(self, parameter1, parameter2):
        self.parameter1 = parameter1
        self.parameter2 = parameter2

    # Method
    def method_name(self):
        return self.parameter1
```

#### Example:
```python
class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def display_info(self):
        return f"Car Brand: {self.brand}, Model: {self.model}"

car1 = Car("Toyota", "Corolla")
print(car1.display_info())
```

### 2. **Object**
An object is an instance of a class. Objects are created using the class blueprint and have attributes and methods defined by the class.

#### Example:
```python
# Using the Car class
car2 = Car("Honda", "Civic")
print(car2.display_info())
```

---

### 3. **Encapsulation**
Encapsulation is the mechanism of wrapping the data (attributes) and methods together. It restricts direct access to some of an object's components, which is achieved using access modifiers.

#### Syntax:
```python
class ClassName:
    def __init__(self):
        self.public_attribute = "Public"
        self._protected_attribute = "Protected"
        self.__private_attribute = "Private"

    def get_private_attribute(self):
        return self.__private_attribute
```

#### Example:
```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Private attribute

    def deposit(self, amount):
        self.__balance += amount

    def get_balance(self):
        return self.__balance

account = BankAccount(1000)
account.deposit(500)
print(account.get_balance())  # Output: 1500
```

---

### 4. **Inheritance**
Inheritance is the process by which one class (child class) can inherit attributes and methods from another class (parent class).

#### Syntax:
```python
class ParentClass:
    # Parent class attributes and methods

class ChildClass(ParentClass):
    # Additional attributes and methods for the child class
```

#### Example:
```python
class Animal:
    def speak(self):
        return "I am an animal"

class Dog(Animal):
    def speak(self):
        return "Woof!"

dog = Dog()
print(dog.speak())  # Output: Woof!
```

---

### 5. **Polymorphism**
Polymorphism allows objects of different classes to be treated as objects of a common parent class. It is often implemented using method overriding or method overloading.

#### Example:
```python
class Bird:
    def sound(self):
        return "Chirp"

class Cat:
    def sound(self):
        return "Meow"

def make_sound(animal):
    print(animal.sound())

make_sound(Bird())  # Output: Chirp
make_sound(Cat())   # Output: Meow
```

---

### 6. **Abstraction**
Abstraction is the process of hiding implementation details and showing only essential features. It can be implemented using abstract base classes in Python.

#### Syntax:
```python
from abc import ABC, abstractmethod

class AbstractClass(ABC):
    @abstractmethod
    def abstract_method(self):
        pass
```

#### Example:
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

rect = Rectangle(10, 5)
print(rect.area())  # Output: 50
```

---

## Key Benefits of OOP

1. **Reusability:** Classes and objects can be reused in multiple programs.
2. **Scalability:** OOP concepts make large-scale projects easier to manage.
3. **Encapsulation:** Ensures security by hiding sensitive data.
4. **Flexibility:** Polymorphism and inheritance provide flexibility in programming.

---

This concludes an overview of the essential OOP concepts in Python. Use these examples to strengthen your understanding and build robust, maintainable programs!
