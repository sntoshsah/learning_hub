# Inheritance in Python

Inheritance is a fundamental concept in Object-Oriented Programming (OOP) that allows a class (child class) to inherit attributes and methods from another class (parent class). This enables code reuse, modularity, and the ability to build upon existing implementations.

---

## Why Use Inheritance?

1. **Code Reusability:** Avoid rewriting common logic by inheriting it from a parent class.
2. **Modularity:** Separate concerns by dividing functionalities among parent and child classes.
3. **Extensibility:** Add or override functionalities in child classes without modifying the parent class.

---

## Types of Inheritance

Python supports the following types of inheritance:

1. **Single Inheritance**
2. **Multiple Inheritance**
3. **Multilevel Inheritance**
4. **Hierarchical Inheritance**
5. **Hybrid Inheritance**

---

## Syntax

```python
class ParentClass:
    # Parent class attributes and methods

class ChildClass(ParentClass):
    # Additional attributes and methods for the child class
```

---

## Examples

### 1. **Single Inheritance**
In single inheritance, a child class inherits from a single parent class.

#### Example:
```python
class Animal:
    def speak(self):
        return "I make sounds"

class Dog(Animal):
    def bark(self):
        return "Woof! Woof!"

# Create an object of the child class
dog = Dog()
print(dog.speak())  # Output: I make sounds
print(dog.bark())   # Output: Woof! Woof!
```

---

### 2. **Multiple Inheritance**
In multiple inheritance, a child class inherits from multiple parent classes.

#### Example:
```python
class Father:
    def skills(self):
        return "Driving"

class Mother:
    def skills(self):
        return "Cooking"

class Child(Father, Mother):
    def all_skills(self):
        return f"{self.skills()} and {Mother.skills(self)}"

# Create an object of the child class
child = Child()
print(child.all_skills())  # Output: Driving and Cooking
```

---

### 3. **Multilevel Inheritance**
In multilevel inheritance, a class inherits from another class, which in turn inherits from another class.

#### Example:
```python
class Vehicle:
    def info(self):
        return "I am a vehicle"

class Car(Vehicle):
    def car_type(self):
        return "I am a car"

class SportsCar(Car):
    def brand(self):
        return "I am a sports car"

# Create an object of the grandchild class
sports_car = SportsCar()
print(sports_car.info())       # Output: I am a vehicle
print(sports_car.car_type())   # Output: I am a car
print(sports_car.brand())      # Output: I am a sports car
```

---

### 4. **Hierarchical Inheritance**
In hierarchical inheritance, multiple child classes inherit from a single parent class.

#### Example:
```python
class Parent:
    def message(self):
        return "This is a message from the parent"

class Child1(Parent):
    def child1_message(self):
        return "This is child 1"

class Child2(Parent):
    def child2_message(self):
        return "This is child 2"

# Create objects of child classes
child1 = Child1()
child2 = Child2()
print(child1.message())        # Output: This is a message from the parent
print(child1.child1_message()) # Output: This is child 1
print(child2.message())        # Output: This is a message from the parent
print(child2.child2_message()) # Output: This is child 2
```

---

### 5. **Hybrid Inheritance**
Hybrid inheritance is a combination of two or more types of inheritance.

#### Example:
```python
class Base:
    def base_message(self):
        return "This is the base class"

class Parent1(Base):
    def parent1_message(self):
        return "This is parent 1"

class Parent2(Base):
    def parent2_message(self):
        return "This is parent 2"

class Child(Parent1, Parent2):
    def child_message(self):
        return "This is the child class"

# Create an object of the child class
child = Child()
print(child.base_message())     # Output: This is the base class
print(child.parent1_message())  # Output: This is parent 1
print(child.parent2_message())  # Output: This is parent 2
print(child.child_message())    # Output: This is the child class
```

---

## Method Overriding in Inheritance

Method overriding allows a child class to provide a specific implementation of a method that is already defined in its parent class.

#### Example:
```python
class Parent:
    def greet(self):
        return "Hello from Parent"

class Child(Parent):
    def greet(self):
        return "Hello from Child"

# Create objects
parent = Parent()
child = Child()
print(parent.greet())  # Output: Hello from Parent
print(child.greet())   # Output: Hello from Child
```

---

## Advantages of Inheritance

1. **Code Reusability:** Eliminates redundancy by reusing common functionality.
2. **Improved Maintainability:** Centralizes common logic, making updates easier.
3. **Extensibility:** Easily extend functionality in child classes.

---

This concludes an in-depth explanation of inheritance in Python with examples and syntax. Use this knowledge to create modular and reusable programs!
