# Encapsulation in Python

Encapsulation is one of the fundamental principles of Object-Oriented Programming (OOP). It is the mechanism of restricting access to certain details of an object and exposing only essential features. This concept helps protect an object’s integrity by preventing unintended interference and misuse of its data.

Encapsulation is implemented in Python by:
1. Defining public, protected, and private attributes.
2. Using getter and setter methods to access and modify private attributes.

---

## **1. Public Attributes**
Public attributes are accessible from anywhere, both inside and outside the class.

#### Syntax:
```python
class ClassName:
    def __init__(self):
        self.public_attribute = "Accessible everywhere"

# Accessing public attribute
obj = ClassName()
print(obj.public_attribute)
```

#### Example:
```python
class Person:
    def __init__(self, name):
        self.name = name  # Public attribute

person = Person("Alice")
print(person.name)  # Output: Alice
```

---

## **2. Protected Attributes**
Protected attributes are indicated by a single underscore (`_`) prefix. They are intended to be used within the class and its subclasses, but can still be accessed directly if needed (not strictly enforced).

#### Syntax:
```python
class ClassName:
    def __init__(self):
        self._protected_attribute = "Accessible within class and subclasses"
```

#### Example:
```python
class Animal:
    def __init__(self, species):
        self._species = species  # Protected attribute

    def get_species(self):
        return self._species

class Dog(Animal):
    def speak(self):
        return f"I am a {self._species}. Woof!"

dog = Dog("Canine")
print(dog.get_species())  # Output: Canine
print(dog.speak())        # Output: I am a Canine. Woof!
```

---

## **3. Private Attributes**
Private attributes are indicated by a double underscore (`__`) prefix. They are only accessible within the class and are not directly accessible from outside the class.

#### Syntax:
```python
class ClassName:
    def __init__(self):
        self.__private_attribute = "Accessible only within the class"
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

# Attempting to access private attribute directly
# print(account.__balance)  # Raises AttributeError
```

---

## **4. Getter and Setter Methods**
Getter and setter methods are used to access and modify private attributes while maintaining control over how the data is manipulated.

#### Syntax:
```python
class ClassName:
    def __init__(self):
        self.__attribute = value

    def get_attribute(self):
        return self.__attribute

    def set_attribute(self, value):
        self.__attribute = value
```

#### Example:
```python
class Employee:
    def __init__(self, salary):
        self.__salary = salary  # Private attribute

    def get_salary(self):
        return self.__salary

    def set_salary(self, value):
        if value > 0:
            self.__salary = value
        else:
            raise ValueError("Salary must be positive")

employee = Employee(5000)
print(employee.get_salary())  # Output: 5000

employee.set_salary(6000)
print(employee.get_salary())  # Output: 6000

# employee.set_salary(-100)  # Raises ValueError
```

---

## **Benefits of Encapsulation**
1. **Data Protection:** Prevents unauthorized access and modification of data.
2. **Modularity:** Makes the class more modular and manageable.
3. **Ease of Maintenance:** Changes to encapsulated code do not affect other parts of the program.
4. **Controlled Access:** Provides controlled access through methods like getters and setters.

Encapsulation ensures the integrity and security of the data while providing the flexibility to modify implementations without affecting external code.
