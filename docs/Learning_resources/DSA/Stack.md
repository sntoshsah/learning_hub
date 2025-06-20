# Stack Data Structure

## Basic Concept of Stack

A stack is a linear data structure that follows the **Last-In-First-Out (LIFO)** principle. This means the last element added to the stack will be the first one to be removed.

### Real-world Analogies:
- Stack of plates (you take the top plate first)
- Browser back button (most recent page visited is the first to go back to)
- Undo functionality in editors (most recent change is undone first)

### Key Characteristics:
- Ordered collection of items
- Addition (push) and removal (pop) happens at the same end (called the "top")
- Limited access - only the top element is accessible

```python
# Visualizing stack operations
stack = []

# Push operations
stack.append(1)  # Stack: [1]
stack.append(2)  # Stack: [1, 2]
stack.append(3)  # Stack: [1, 2, 3]

# Pop operations
print(stack.pop())  # Output: 3, Stack: [1, 2]
print(stack.pop())  # Output: 2, Stack: [1]
print(stack.pop())  # Output: 1, Stack: []
```

## Stack as an Abstract Data Type (ADT)

As an ADT, a stack is defined by its behavior rather than its implementation. The stack ADT specifies:

### Main Operations:
1. **push(item)**: Add an item to the top of the stack
2. **pop()**: Remove and return the top item
3. **peek()/top()**: Return the top item without removing it
4. **is_empty()**: Check if the stack is empty
5. **size()**: Return the number of items in the stack

```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add an item to the top of the stack"""
        self.items.append(item)
    
    def pop(self):
        """Remove and return the top item"""
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("pop from empty stack")
    
    def peek(self):
        """Return the top item without removing it"""
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("peek from empty stack")
    
    def is_empty(self):
        """Check if the stack is empty"""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the stack"""
        return len(self.items)
    
    def __str__(self):
        return str(self.items)

# Usage example
s = Stack()
s.push(10)
s.push(20)
s.push(30)
print(f"Stack: {s}")          # Output: Stack: [10, 20, 30]
print(f"Top item: {s.peek()}") # Output: Top item: 30
print(f"Popped: {s.pop()}")    # Output: Popped: 30
print(f"Stack size: {s.size()}") # Output: Stack size: 2
```

## Stack Operations

### Time Complexities:
- **push()**: O(1) - constant time
- **pop()**: O(1) - constant time
- **peek()**: O(1) - constant time
- **is_empty()**: O(1) - constant time
- **size()**: O(1) - constant time

### Implementation Variations:
1. **Using Lists**: Python lists can be used directly (as shown above)
2. **Using collections.deque**: More efficient for large stacks
3. **Using Linked List**: Better control over memory

```python
# Stack implementation using collections.deque
from collections import deque

class DequeStack:
    def __init__(self):
        self.items = deque()
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop()
    
    def peek(self):
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Linked List Node for Stack
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

# Stack implementation using Linked List
class LinkedListStack:
    def __init__(self):
        self.top = None
        self._size = 0
    
    def push(self, item):
        new_node = Node(item)
        new_node.next = self.top
        self.top = new_node
        self._size += 1
    
    def pop(self):
        if self.top is None:
            raise IndexError("pop from empty stack")
        item = self.top.data
        self.top = self.top.next
        self._size -= 1
        return item
    
    def peek(self):
        if self.top is None:
            raise IndexError("peek from empty stack")
        return self.top.data
    
    def is_empty(self):
        return self.top is None
    
    def size(self):
        return self._size
```

## Stack Applications

### Common Use Cases:
1. **Function call management** (call stack)
2. **Undo/Redo operations** in editors
3. **Browser history** management
4. **Backtracking algorithms** (maze solving, etc.)
5. **Expression evaluation** and syntax parsing
6. **Memory management** in some systems

```python
# Example: Balanced Parentheses Checker
def is_balanced(expr):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in expr:
        if char in mapping.values():
            stack.append(char)
        elif char in mapping.keys():
            if not stack or stack.pop() != mapping[char]:
                return False
    return not stack

print(is_balanced("({[]})"))  # True
print(is_balanced("({[}])"))  # False

# Example: Browser Back/Forward Navigation
class Browser:
    def __init__(self):
        self.back_stack = []
        self.forward_stack = []
        self.current_page = None
    
    def visit(self, url):
        if self.current_page:
            self.back_stack.append(self.current_page)
        self.current_page = url
        self.forward_stack = []
    
    def back(self):
        if self.back_stack:
            self.forward_stack.append(self.current_page)
            self.current_page = self.back_stack.pop()
            return self.current_page
        return None
    
    def forward(self):
        if self.forward_stack:
            self.back_stack.append(self.current_page)
            self.current_page = self.forward_stack.pop()
            return self.current_page
        return None

# Usage
browser = Browser()
browser.visit("google.com")
browser.visit("github.com")
browser.visit("python.org")
print(browser.back())    # Output: github.com
print(browser.back())    # Output: google.com
print(browser.forward()) # Output: github.com
```

## Conversion from Infix to Postfix Expressions

Infix notation is the common arithmetic notation (e.g., A + B), while postfix notation (Reverse Polish Notation) places the operator after its operands (e.g., A B +).

### Algorithm Steps:
1. Initialize an empty stack and empty output list
2. Scan the infix expression from left to right
3. If operand, add to output
4. If '(', push to stack
5. If ')', pop from stack and add to output until '(' is encountered
6. If operator:
   - While stack not empty and top has higher/equal precedence
   - Pop operator from stack to output
   - Push current operator to stack
7. Pop any remaining operators from stack to output

```python
def infix_to_postfix(infix_expr):
    precedence = {'^': 4, '*': 3, '/': 3, '+': 2, '-': 2}
    stack = []
    output = []
    
    for char in infix_expr:
        if char.isalnum():  # Operand
            output.append(char)
        elif char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # Remove '(' from stack
        else:  # Operator
            while (stack and stack[-1] != '(' and
                   precedence.get(char, 0) <= precedence.get(stack[-1], 0)):
                output.append(stack.pop())
            stack.append(char)
    
    while stack:
        output.append(stack.pop())
    
    return ''.join(output)

# Examples
print(infix_to_postfix("A+B*C"))      # Output: ABC*+
print(infix_to_postfix("(A+B)*C"))    # Output: AB+C*
print(infix_to_postfix("A+B*(C^D-E)")) # Output: ABCD^E-*+
```

## Evaluation of Postfix Expressions

Postfix evaluation is simpler than infix as it doesn't need parentheses or precedence rules.

### Algorithm Steps:
1. Initialize an empty stack
2. Scan the postfix expression from left to right
3. If operand, push to stack
4. If operator, pop top two operands, apply operator, push result
5. After scanning, the stack should contain exactly one element (the result)

```python
def evaluate_postfix(postfix_expr):
    stack = []
    
    for char in postfix_expr:
        if char.isdigit():
            stack.append(int(char))
        else:
            operand2 = stack.pop()
            operand1 = stack.pop()
            result = apply_operator(operand1, operand2, char)
            stack.append(result)
    
    return stack.pop()

def apply_operator(a, b, operator):
    if operator == '+':
        return a + b
    elif operator == '-':
        return a - b
    elif operator == '*':
        return a * b
    elif operator == '/':
        return a / b
    elif operator == '^':
        return a ** b
    else:
        raise ValueError("Unsupported operator")

# Examples
print(evaluate_postfix("23*5+"))    # Output: 11 (2*3 + 5)
print(evaluate_postfix("345*+6-"))  # Output: 17 (3 + 4*5 - 6)
print(evaluate_postfix("52^3*"))    # Output: 75 (5^2 * 3)

# Combined example: Convert infix to postfix and evaluate
infix_expr = "3+4*5-6"
postfix_expr = infix_to_postfix(infix_expr)
result = evaluate_postfix(postfix_expr)
print(f"Infix: {infix_expr} → Postfix: {postfix_expr} → Result: {result}")
# Output: Infix: 3+4*5-6 → Postfix: 345*+6- → Result: 17
```

### Handling Multi-digit Numbers:
The basic implementation above handles single-digit numbers. Here's an enhanced version for multi-digit numbers:

```python
def enhanced_infix_to_postfix(infix_expr):
    precedence = {'^': 4, '*': 3, '/': 3, '+': 2, '-': 2}
    stack = []
    output = []
    i = 0
    n = len(infix_expr)
    
    while i < n:
        char = infix_expr[i]
        
        if char == ' ':
            i += 1
            continue
        
        # Handle multi-digit numbers
        if char.isdigit():
            num = []
            while i < n and infix_expr[i].isdigit():
                num.append(infix_expr[i])
                i += 1
            output.append(''.join(num))
            continue
        
        if char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # Remove '('
        else:  # Operator
            while (stack and stack[-1] != '(' and
                   precedence.get(char, 0) <= precedence.get(stack[-1], 0)):
                output.append(stack.pop())
            stack.append(char)
        
        i += 1
    
    while stack:
        output.append(stack.pop())
    
    return ' '.join(output)

def enhanced_evaluate_postfix(postfix_expr):
    stack = []
    tokens = postfix_expr.split()
    
    for token in tokens:
        if token.isdigit():
            stack.append(int(token))
        else:
            operand2 = stack.pop()
            operand1 = stack.pop()
            result = apply_operator(operand1, operand2, token)
            stack.append(result)
    
    return stack.pop()

# Example with multi-digit numbers
infix_expr = "10 + 20 * (30 / 5) - 4"
postfix_expr = enhanced_infix_to_postfix(infix_expr)
result = enhanced_evaluate_postfix(postfix_expr)
print(f"Infix: {infix_expr} → Postfix: {postfix_expr} → Result: {result}")
# Output: Infix: 10 + 20 * (30 / 5) - 4 → Postfix: 10 20 30 5 / * + 4 - → Result: 126
```