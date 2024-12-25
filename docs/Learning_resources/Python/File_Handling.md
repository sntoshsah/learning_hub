# Python File Handling and File I/O

File handling in Python is a vital skill for working with external files to read, write, and manage data. Python provides a simple and intuitive interface for file operations through built-in functions and methods.

---

## Why File Handling is Important

File handling allows programs to:
- Store data persistently.
- Share data between programs.
- Work with large datasets efficiently.

---

## Basic File Operations

### Opening a File
The `open()` function is used to open a file. It returns a file object and supports the following modes:

| Mode | Description                          |
|------|--------------------------------------|
| `r`  | Read mode (default).                 |
| `w`  | Write mode. Creates a file if it doesn't exist. Overwrites existing content. |
| `x`  | Exclusive creation. Fails if the file exists. |
| `a`  | Append mode. Creates the file if it doesn't exist. |
| `b`  | Binary mode.                          |
| `t`  | Text mode (default).                 |
| `+`  | Open for both reading and writing.   |

#### Syntax
```python
file = open("filename", "mode")
```

#### Example
```python
# Open a file in read mode
data_file = open("data.txt", "r")
print(data_file.read())
data_file.close()
```

---

### Reading a File
Python provides multiple methods to read from a file.

- `read(size)`: Reads the specified number of bytes. Reads the entire file if no size is provided.
- `readline()`: Reads a single line from the file.
- `readlines()`: Reads all lines and returns them as a list.

#### Example
```python
# Reading a file
with open("data.txt", "r") as file:
    content = file.read()
    print(content)

# Reading line by line
with open("data.txt", "r") as file:
    for line in file:
        print(line.strip())
```

---

### Writing to a File
To write data to a file, use the `write()` or `writelines()` methods.

- `write(string)`: Writes a string to the file.
- `writelines(list)`: Writes a list of strings to the file.

#### Example
```python
# Writing to a file
with open("output.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("Python File I/O is simple.")

# Writing multiple lines
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open("output.txt", "w") as file:
    file.writelines(lines)
```

---

### Appending to a File
Use the `a` mode to append data to a file without overwriting its existing content.

#### Example
```python
# Appending data to a file
with open("output.txt", "a") as file:
    file.write("This is an appended line.\n")
```

---

### Closing a File
Always close a file after operations to free up system resources. This can be done explicitly using `file.close()` or automatically with a `with` statement.

#### Example
```python
# Explicitly closing a file
file = open("data.txt", "r")
print(file.read())
file.close()

# Using a with statement
with open("data.txt", "r") as file:
    print(file.read())  # File is automatically closed after this block
```

---

## File Positioning
The `seek()` and `tell()` methods help navigate through a file.

- `tell()`: Returns the current file pointer position.
- `seek(offset, from_what)`: Moves the file pointer to the specified position.
  - `from_what` values: `0` (default, beginning of file), `1` (current position), `2` (end of file).

#### Example
```python
with open("data.txt", "r") as file:
    print(file.read(5))      # Read first 5 bytes
    print(file.tell())       # Print current position
    file.seek(0)             # Move pointer to the beginning
    print(file.read(5))      # Read first 5 bytes again
```

---

## Working with Binary Files
Binary files store data in a non-human-readable format (e.g., images, videos).

#### Example
```python
# Writing binary data
with open("image.png", "rb") as source:
    content = source.read()

with open("copy.png", "wb") as destination:
    destination.write(content)
```

---

## File Deletion
Use the `os` module to delete files.

#### Example
```python
import os

# Deleting a file
if os.path.exists("output.txt"):
    os.remove("output.txt")
    print("File deleted.")
else:
    print("File does not exist.")
```

---

## Exception Handling in File Operations
Use `try-except` blocks to handle errors gracefully.

#### Example
```python
try:
    with open("non_existent.txt", "r") as file:
        print(file.read())
except FileNotFoundError:
    print("File not found!")
```

---

## Summary of File I/O Methods

| Method         | Description                                     |
|----------------|-------------------------------------------------|
| `open()`       | Opens a file.                                  |
| `read()`       | Reads the entire file or specified bytes.      |
| `readline()`   | Reads a single line.                           |
| `readlines()`  | Reads all lines as a list.                     |
| `write()`      | Writes a string to the file.                   |
| `writelines()` | Writes a list of strings to the file.          |
| `close()`      | Closes the file.                               |
| `seek()`       | Moves the file pointer to a specific position. |
| `tell()`       | Returns the current file pointer position.     |

---

File handling is a fundamental skill in Python, enabling programs to interact with external files effectively. Understanding these concepts will help you manage data efficiently in real-world applications.

