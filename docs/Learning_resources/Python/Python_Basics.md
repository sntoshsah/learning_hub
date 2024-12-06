## Python Basics: Variables, Datatypes, Keywords, and Literals

This guide provides a beginner-friendly introduction to variables, datatypes, keywords, and literals in Python, with examples and explanations.<br>
**1. Variables**<br>
Definition

Variables are containers for storing data values.
In Python, you don’t need to declare a variable type explicitly. The type is inferred when you assign a value.

Syntax:

variable_name = value

Rules

    Must start with a letter or an underscore _.
    Cannot start with a number.
    Can contain letters, numbers, and underscores.
    Case-sensitive (name and Name are different).

Example:
```python exec='off'
# Assigning values to variables
x = 10           # Integer
name = "Alice"   # String
_pi = 3.14       # Float
is_valid = True  # Boolean

# Printing variables
print(x, name, _pi, is_valid)
```
```python exec='on'
# Assigning values to variables
x = 10           # Integer
name = "Alice"   # String
_pi = 3.14       # Float
is_valid = True  # Boolean

# Printing variables
print(x, name, _pi, is_valid)
```
**2. Datatypes**<br>
Definition

Datatypes define the type of data a variable can hold.<br>

Common Datatypes

| Datatype | Description                       | Example                                |
|----------|-----------------------------------|----------------------------------------|
| `int`    | Integer numbers                  | `x = 5`                               |
| `float`  | Decimal numbers                  | `y = 3.14`                            |
| `str`    | Text or string data              | `name = "Python"`                     |
| `bool`   | True/False values                | `is_valid = True`                     |
| `list`   | Ordered collection of items      | `nums = [1, 2, 3]`                    |
| `tuple`  | Immutable ordered collection     | `coords = (10, 20)`                   |
| `dict`   | Key-value pairs                  | `person = {"name": "Alice", "age": 25}` |
| `set`    | Unordered collection of unique items | `unique_nums = {1, 2, 3}`              |

Examples
```python 
#### Integer
age = 25
print(type(age))  # Output: <class 'int'>

#### Float
price = 19.99
print(type(price))  # Output: <class 'float'>

#### String
greeting = "Hello, World!"
print(type(greeting))  # Output: <class 'str'>

#### List
colors = ["red", "green", "blue"]
print(type(colors))  # Output: <class 'list'>

#### Dictionary
person = {"name": "Alice", "age": 30}
print(type(person))  # Output: <class 'dict'>
```

**3. Keywords**<br>

Definition<br>
Keywords are reserved words in Python.
They have a special meaning and cannot be used as variable names.

Examples of Keywords

and, or, not, if, else, while, for, import, def, class, True, False, None, return, etc.
Usage
```python
#### Using keywords
if True:
    print("This is a keyword example.")

Check All Keywords

import keyword
print(keyword.kwlist)
```

**4. Literals**<br>
Definition

Literals are fixed values assigned to variables or used directly in expressions.

Types of Literals

| Literal Type | Description          	    | Example |
|--------------|----------------------------|-----------|
| Numeric      | Integer, Float, Complex 	| `10, 3.14, 2 + 3j` |
| String	   | Text data               	| `"Hello", 'Python'` |
| Boolean      | Logical True/False Values  | `True, False`|
| Special      | Represents absence of value| `None` |

Examples
```python
#### Numeric Literals
int_literal = 100
float_literal = 20.5
complex_literal = 3 + 4j

#### String Literals
single_quote = 'Hello'
double_quote = "World"

#### Boolean Literals
is_active = True

#### Special Literal
empty_value = None

print(int_literal, float_literal, complex_literal, single_quote, double_quote, is_active, empty_value)
```
Output:
```python exec="on"
#### Numeric Literals
int_literal = 100
float_literal = 20.5
complex_literal = 3 + 4j

#### String Literals
single_quote = 'Hello'
double_quote = "World"

#### Boolean Literals
is_active = True

#### Special Literal
empty_value = None

print(int_literal, float_literal, complex_literal, single_quote, double_quote, is_active, empty_value)
```

**5. Python is Dynamically Typed**

Python doesn’t require you to specify the type of a variable. The type is determined automatically based on the value assigned.
Example
```python
x = 10         # x is an integer
x = "Python"   # Now x is a string
print(x)       # Output: Python
```
**6. Example Program**
```python
# Define variables
name = "Alice"
age = 30
is_student = False
grades = [85, 90, 92]

# Display information
print(f"Name: {name}\n")
print(f"Age: {age}\n")
print(f"Is Student: {is_student}\n")
print(f"Grades: {grades}")
```
Output:
```python exec='on'
# Define variables
name = "Alice"
age = 30
is_student = False
grades = [85, 90, 92]

# Display information
print(f"Name: {name}\n")
print(f"Age: {age}\n")
print(f"Is Student: {is_student}\n")
print(f"Grades: {grades}")
```

**Quick Tips**

    Use meaningful variable names (e.g., price, age, username).
    Use type() to check the datatype of a variable.
    Use comments (#) to explain your code.

This guide provides the foundation for understanding Python's core concepts. Keep practicing, and soon you'll master them! 😊