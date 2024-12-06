## First Python Program
```python
print("Hello World!")
```
Output:
```python exec='on'
print("Hello World!")
```
# Basic Syntax of Python
**Definition:**
Python syntax refers to the set of rules that defines how Python code is written and interpreted. Python is designed to be simple, clear, and easy to read. Unlike some programming languages, Python emphasizes readability and uses whitespace to structure code.

Key Features of Python Syntax:

1. No need for semicolons: Statements end with a newline by default, though semicolons can be used to separate multiple statements on a single line.
2. Case-sensitive: Python differentiates between uppercase and lowercase letters (e.g., Variable and variable are distinct).
3. Indentation: Used to define blocks of code, such as in loops, conditionals, or functions (explained further below).
4. Comments: Start with # for single-line comments and triple quotes (''' or """) for multi-line comments.
<br>  
### Example of Python Syntax
```python
# Assign values to variables
x = 10  
y = 20

# Conditional statement
if x < y:
    print("x is less than y")
else:
    print("x is not less than y")

```


## Comments in Python

**Definition:**
A comment is a line of text in Python code that is not executed by the interpreter. Comments are used to explain the code, make it more readable, or temporarily disable certain parts of the code.

Python provides two types of comments:

1. Single-line comments
2. Multi-line comments

**Single-line Comments**

Single-line comments start with a # symbol. Any text after the # on the same line is treated as a comment and is ignored by the Python interpreter.
Example:
```python 
# This is a single-line comment
x = 10  # This is an inline comment
print(x)  # Printing the value of x
```
Explanation:

    The # symbol makes the text after it a comment.
    In this example:
        The first line explains the purpose of the code.
        The inline comments next to x = 10 and print(x) provide additional context.



**Multi-line Comments**

Python does not have a specific multi-line comment syntax like other languages. However, multi-line comments are usually implemented using a block of strings enclosed in triple quotes (''' or """). While not technically comments, such strings are ignored by the interpreter if they are not assigned to a variable or used as a docstring.
Example:
```python
'''
This is a multi-line comment.
It can span multiple lines
and is useful for providing detailed explanations.
'''

x = 10
y = 20

"""
This part of the code adds two numbers.
You can use triple double-quotes as well for multi-line comments.
"""

result = x + y
print(result)  # Printing the sum
```
Explanation:

    The triple quotes allow you to write a block of text, making it useful for longer explanations or documentation.



**Practical Use**
```python
# Define a function to add two numbers

def add_numbers(a, b):
    """
    This function takes two arguments, 'a' and 'b'.
    It returns the sum of these numbers.
    """
    return a + b

# Call the function
sum_result = add_numbers(5, 7)
print(sum_result)  # Outputs: 12
```
Explanation:

    The single-line comment explains the function definition.
    The multi-line comment (docstring) describes what the function does.

## Python Indentation

**Definition:**
Indentation refers to the spaces at the beginning of a code line. Unlike many other programming languages that use braces {} to define code blocks, Python uses indentation to group statements and define the structure of the program. Consistent indentation is mandatory in Python, as improper indentation will lead to a syntax error.
Rules for Indentation:

1. Indentation must be consistent within the same block.
2. The default indentation level is typically 4 spaces (though tabs or other levels are allowed if used consistently).
<br>Indentation is required for:
    * Loops (for, while)
    * Conditionals (if, else, elif)
    * Function definitions
    * Class definitions

Example of Indentation
```python
# Function with proper indentation
def greet(name):
    # This line is indented
    print(f"Hello, {name}!")

# Using the function
greet("Alice")

# Conditional statement with indentation
age = 18
if age >= 18:
    print("You are eligible to vote.")
else:
    print("You are not eligible to vote.")
```
Incorrect Indentation Example
``` python
def greet(name):
print(f"Hello, {name}!")  # IndentationError: expected an indented block
```
Output:
``` python exec='on'
def greet(name):
print(f"Hello, {name}!")  # IndentationError: expected an indented block
```