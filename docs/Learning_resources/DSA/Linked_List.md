# Lists and Linked Lists in Python

## Basic Concept of Lists

A list is an ordered collection of elements that allows for dynamic sizing and various operations. In Python, lists are implemented as dynamic arrays.

### Characteristics:
- Ordered sequence of elements
- Mutable (can be modified after creation)
- Can contain elements of different types
- Zero-based indexing
- Dynamic resizing (grows as needed)

```python
# Creating and using lists
my_list = [1, 2, 3, 'four', 5.0]
print(my_list[0])      # 1 (indexing)
print(my_list[-1])     # 5.0 (negative indexing)
my_list.append(6)      # Add to end
my_list.insert(2, 2.5) # Insert at position
print(len(my_list))    # 7 (length)
```

## List as an Abstract Data Type (ADT)

The List ADT defines the following operations independent of implementation:

### Core Operations:
1. **create()**: Initialize an empty list
2. **insert(pos, item)**: Add item at position
3. **delete(pos)**: Remove item at position
4. **get(pos)**: Retrieve item at position
5. **size()**: Return number of elements
6. **is_empty()**: Check if list is empty
7. **replace(pos, item)**: Change item at position

```python
class ListADT:
    def __init__(self):
        self.items = []
    
    def insert(self, pos, item):
        self.items.insert(pos, item)
    
    def delete(self, pos):
        return self.items.pop(pos)
    
    def get(self, pos):
        return self.items[pos]
    
    def size(self):
        return len(self.items)
    
    def is_empty(self):
        return len(self.items) == 0
    
    def replace(self, pos, item):
        self.items[pos] = item
    
    def __str__(self):
        return str(self.items)

# Usage
adt_list = ListADT()
adt_list.insert(0, 10)
adt_list.insert(1, 20)
print(adt_list)  # [10, 20]
```

## Array Implementation of Lists

Python lists are implemented as dynamic arrays that:
- Allocate contiguous memory
- Automatically resize when capacity is exceeded
- Provide amortized O(1) append operations

```python
# Simplified array-based list implementation
class ArrayList:
    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.array = self._make_array(self.capacity)
    
    def _make_array(self, new_capacity):
        return [None] * new_capacity
    
    def _resize(self, new_capacity):
        new_array = self._make_array(new_capacity)
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array
        self.capacity = new_capacity
    
    def append(self, item):
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        self.array[self.size] = item
        self.size += 1
    
    def __getitem__(self, index):
        if 0 <= index < self.size:
            return self.array[index]
        raise IndexError('Index out of bounds')
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        return str(self.array[:self.size])

# Usage
arr_list = ArrayList()
arr_list.append(1)
arr_list.append(2)
print(arr_list)  # [1, 2]
```

## Linked List Introduction

A linked list is a linear data structure where elements are stored in nodes, and each node points to the next node.

### Advantages over arrays:
- Dynamic size
- Efficient insertions/deletions
- No memory waste (allocates exactly what's needed)

### Disadvantages:
- No random access
- Extra memory for pointers
- Cache locality is worse than arrays

## Types of Linked Lists

### 1. Singly Linked List
Each node contains data and a pointer to the next node.

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node
    
    def display(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

# Usage
sll = SinglyLinkedList()
sll.append(1)
sll.append(2)
sll.append(3)
sll.display()  # 1 -> 2 -> 3 -> None
```

### 2. Doubly Linked List
Each node contains data and pointers to both next and previous nodes.

```python
class DoublyNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = DoublyNode(data)
        if not self.head:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node
        new_node.prev = last
    
    def display(self):
        current = self.head
        while current:
            print(current.data, end=" <-> ")
            current = current.next
        print("None")

# Usage
dll = DoublyLinkedList()
dll.append(1)
dll.append(2)
dll.append(3)
dll.display()  # 1 <-> 2 <-> 3 <-> None
```

### 3. Circular Linked List
The last node points back to the first node (can be singly or doubly linked).

```python
class CircularLinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            new_node.next = self.head
            return
        last = self.head
        while last.next != self.head:
            last = last.next
        last.next = new_node
        new_node.next = self.head
    
    def display(self):
        if not self.head:
            return
        current = self.head
        while True:
            print(current.data, end=" -> ")
            current = current.next
            if current == self.head:
                break
        print("HEAD")

# Usage
cll = CircularLinkedList()
cll.append(1)
cll.append(2)
cll.append(3)
cll.display()  # 1 -> 2 -> 3 -> HEAD
```

## Basic Linked List Operations

### Node Creation
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
```

### Insertion Operations
1. **At beginning**
2. **At end**
3. **After a given node**

```python
class LinkedList:
    def __init__(self):
        self.head = None
    
    def insert_at_beginning(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def insert_at_end(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node
    
    def insert_after(self, prev_node, data):
        if not prev_node:
            print("Previous node must be in the list")
            return
        new_node = Node(data)
        new_node.next = prev_node.next
        prev_node.next = new_node
```

### Deletion Operations
1. **By key** (value)
2. **By position**

```python
    def delete_node(self, key):
        temp = self.head
        
        # If head node itself holds the key
        if temp and temp.data == key:
            self.head = temp.next
            temp = None
            return
        
        # Search for the key
        prev = None
        while temp and temp.data != key:
            prev = temp
            temp = temp.next
        
        # If key not present
        if not temp:
            return
        
        # Unlink the node
        prev.next = temp.next
        temp = None
    
    def delete_at_position(self, pos):
        if not self.head:
            return
        
        temp = self.head
        
        # If head needs to be removed
        if pos == 0:
            self.head = temp.next
            temp = None
            return
        
        # Find previous node of the node to be deleted
        for i in range(pos-1):
            temp = temp.next
            if not temp:
                break
        
        # If position is more than number of nodes
        if not temp or not temp.next:
            return
        
        # Node temp.next is the node to be deleted
        next_node = temp.next.next
        temp.next = None
        temp.next = next_node
```

## Stack as Linked List

```python
class StackNode:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedStack:
    def __init__(self):
        self.top = None
    
    def push(self, data):
        new_node = StackNode(data)
        new_node.next = self.top
        self.top = new_node
    
    def pop(self):
        if not self.top:
            return None
        popped = self.top.data
        self.top = self.top.next
        return popped
    
    def peek(self):
        return self.top.data if self.top else None
    
    def is_empty(self):
        return self.top is None

# Usage
stack = LinkedStack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())  # 3
print(stack.peek()) # 2
```

## Queue as Linked List

```python
class QueueNode:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedQueue:
    def __init__(self):
        self.front = None
        self.rear = None
    
    def enqueue(self, data):
        new_node = QueueNode(data)
        if not self.rear:
            self.front = self.rear = new_node
            return
        self.rear.next = new_node
        self.rear = new_node
    
    def dequeue(self):
        if not self.front:
            return None
        temp = self.front
        self.front = temp.next
        if not self.front:
            self.rear = None
        return temp.data
    
    def is_empty(self):
        return self.front is None

# Usage
queue = LinkedQueue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue.dequeue())  # 1
print(queue.dequeue())  # 2
```