# Queue Data Structure

## Basic Concept of Queue

A queue is a linear data structure that follows the **First-In-First-Out (FIFO)** principle. This means the first element added to the queue will be the first one to be removed.

### Real-world Analogies:
- Line at a ticket counter (first person in line gets served first)
- Printer job queue (first document sent gets printed first)
- Customer service calls (first caller gets served first)

### Key Characteristics:
- Ordered collection of items
- Addition (enqueue) happens at the "rear"
- Removal (dequeue) happens at the "front"
- Limited access - only front and rear elements are directly accessible

```python
# Visualizing queue operations using a list (not efficient for large queues)
queue = []

# Enqueue operations
queue.append(1)  # Queue: [1]
queue.append(2)  # Queue: [1, 2]
queue.append(3)  # Queue: [1, 2, 3]

# Dequeue operations
print(queue.pop(0))  # Output: 1, Queue: [2, 3]
print(queue.pop(0))  # Output: 2, Queue: [3]
print(queue.pop(0))  # Output: 3, Queue: []
```

## Queue as an Abstract Data Type (ADT)

As an ADT, a queue is defined by its behavior rather than its implementation. The queue ADT specifies:

### Main Operations:
1. **enqueue(item)**: Add an item to the rear of the queue
2. **dequeue()**: Remove and return the front item
3. **front()/peek()**: Return the front item without removing it
4. **is_empty()**: Check if the queue is empty
5. **size()**: Return the number of items in the queue

```python
class Queue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        """Add an item to the rear of the queue"""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return the front item"""
        if not self.is_empty():
            return self.items.pop(0)
        raise IndexError("dequeue from empty queue")
    
    def front(self):
        """Return the front item without removing it"""
        if not self.is_empty():
            return self.items[0]
        raise IndexError("front from empty queue")
    
    def is_empty(self):
        """Check if the queue is empty"""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the queue"""
        return len(self.items)
    
    def __str__(self):
        return str(self.items)

# Usage example
q = Queue()
q.enqueue(10)
q.enqueue(20)
q.enqueue(30)
print(f"Queue: {q}")          # Output: Queue: [10, 20, 30]
print(f"Front item: {q.front()}") # Output: Front item: 10
print(f"Dequeued: {q.dequeue()}")    # Output: Dequeued: 10
print(f"Queue size: {q.size()}") # Output: Queue size: 2
```

## Primitive Operations in Queue

### Time Complexities (for list implementation):
- **enqueue()**: O(1) - constant time (amortized)
- **dequeue()**: O(n) - linear time (because we use pop(0))
- **front()**: O(1) - constant time
- **is_empty()**: O(1) - constant time
- **size()**: O(1) - constant time

### More Efficient Implementations:
1. **Using collections.deque**: Efficient for both enqueue and dequeue
2. **Using Linked List**: Constant time for all operations

```python
# Queue implementation using collections.deque
from collections import deque

class DequeQueue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        return self.items.popleft()
    
    def front(self):
        return self.items[0]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Linked List Node for Queue
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

# Queue implementation using Linked List
class LinkedListQueue:
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0
    
    def enqueue(self, item):
        new_node = Node(item)
        if self.rear is None:
            self.front = self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node
        self._size += 1
    
    def dequeue(self):
        if self.front is None:
            raise IndexError("dequeue from empty queue")
        item = self.front.data
        self.front = self.front.next
        if self.front is None:
            self.rear = None
        self._size -= 1
        return item
    
    def peek(self):
        if self.front is None:
            raise IndexError("peek from empty queue")
        return self.front.data
    
    def is_empty(self):
        return self.front is None
    
    def size(self):
        return self._size
```

## Linear Queue

A linear queue is the simplest form where elements are added at the rear and removed from the front in a linear manner.

### Limitations:
- Fixed size (in array implementation)
- Inefficient space utilization (can't reuse empty spaces after dequeue)

```python
class LinearQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = 0
        self.rear = -1
        self.size = 0
    
    def enqueue(self, item):
        if self.is_full():
            raise OverflowError("Queue is full")
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.size += 1
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        item = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.queue[self.front]
    
    def is_empty(self):
        return self.size == 0
    
    def is_full(self):
        return self.size == self.capacity
    
    def __str__(self):
        if self.is_empty():
            return "[]"
        if self.front <= self.rear:
            return str(self.queue[self.front:self.rear+1])
        else:
            return str(self.queue[self.front:] + self.queue[:self.rear+1])

# Usage
lq = LinearQueue(5)
lq.enqueue(10)
lq.enqueue(20)
lq.enqueue(30)
print(lq.dequeue())  # 10
lq.enqueue(40)
lq.enqueue(50)
print(lq)  # [20, 30, 40, 50]
```

## Circular Queue

A circular queue improves upon the linear queue by reusing empty spaces in a circular manner.

### Advantages:
- Better space utilization
- Fixed size but reuses empty spaces

```python
class CircularQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = 0
        self.rear = -1
        self.size = 0
    
    def enqueue(self, item):
        if self.is_full():
            raise OverflowError("Queue is full")
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.size += 1
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        item = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.queue[self.front]
    
    def is_empty(self):
        return self.size == 0
    
    def is_full(self):
        return self.size == self.capacity
    
    def __str__(self):
        if self.is_empty():
            return "[]"
        items = []
        for i in range(self.size):
            index = (self.front + i) % self.capacity
            items.append(str(self.queue[index]))
        return "[" + ", ".join(items) + "]"

# Usage
cq = CircularQueue(3)
cq.enqueue(10)
cq.enqueue(20)
cq.enqueue(30)
print(cq.dequeue())  # 10
cq.enqueue(40)  # Works because we've dequeued one item
print(cq)  # [20, 30, 40]
```

## Priority Queue

A priority queue is a special type of queue where each element has a priority, and elements are dequeued based on priority rather than insertion order.

### Characteristics:
- Higher priority elements are dequeued first
- Elements with same priority are dequeued in FIFO order
- Typically implemented using heaps for efficiency

```python
# Implementation using heapq module
import heapq

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0  # To handle items with same priority
    
    def enqueue(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        return heapq.heappop(self._queue)[-1]  # Return the item only
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        return self._queue[0][-1]
    
    def is_empty(self):
        return len(self._queue) == 0
    
    def size(self):
        return len(self._queue)

# Usage
pq = PriorityQueue()
pq.enqueue("Task 1", 3)
pq.enqueue("Task 2", 1)
pq.enqueue("Task 3", 2)
print(pq.dequeue())  # Task 2 (highest priority)
print(pq.dequeue())  # Task 3
print(pq.dequeue())  # Task 1

# Implementation for FIFO same-priority items
class PriorityQueueFIFO:
    def __init__(self):
        self._queue = []
        self._counter = 0
    
    def enqueue(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._counter, item))
        self._counter += 1
    
    def dequeue(self):
        return heapq.heappop(self._queue)[-1]
    
    # Other methods same as above

# Usage with same priority
pqf = PriorityQueueFIFO()
pqf.enqueue("Task A", 1)
pqf.enqueue("Task B", 1)
print(pqf.dequeue())  # Task A (same priority, but enqueued first)
print(pqf.dequeue())  # Task B
```

## Queue Applications

### Common Use Cases:
1. **CPU Scheduling** (process scheduling)
2. **Disk Scheduling** (I/O request handling)
3. **Breadth-First Search** (BFS) in graphs
4. **Print spooling** (managing print jobs)
5. **Call center systems** (handling incoming calls)
6. **Network packet routing** (handling data packets)

```python
# Example: Breadth-First Search (BFS) using Queue
def bfs(graph, start):
    visited = set()
    queue = Queue()
    queue.enqueue(start)
    visited.add(start)
    
    while not queue.is_empty():
        vertex = queue.dequeue()
        print(vertex, end=" ")
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.enqueue(neighbor)

# Example graph
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print("BFS Traversal:")
bfs(graph, 'A')  # Output: A B C D E F

# Example: Printer Job Management
class PrinterQueue:
    def __init__(self):
        self.queue = Queue()
    
    def add_job(self, document, priority=0):
        self.queue.enqueue((document, priority))
    
    def print_job(self):
        if self.queue.is_empty():
            print("No jobs to print")
            return
        document, priority = self.queue.dequeue()
        print(f"Printing: {document} (Priority: {priority})")
    
    def job_count(self):
        return self.queue.size()

# Usage
printer = PrinterQueue()
printer.add_job("Report.pdf", 1)
printer.add_job("Presentation.pptx", 2)
printer.add_job("Image.jpg")
printer.print_job()  # Printing: Report.pdf (Priority: 1)
printer.print_job()  # Printing: Presentation.pptx (Priority: 2)
printer.print_job()  # Printing: Image.jpg (Priority: 0)

# Example: Ticket Counter Simulation
import random
import time

class TicketCounter:
    def __init__(self):
        self.queue = Queue()
        self.ticket_number = 0
    
    def new_customer(self):
        self.ticket_number += 1
        self.queue.enqueue(self.ticket_number)
        print(f"Customer {self.ticket_number} joined the queue")
    
    def serve_customer(self):
        if self.queue.is_empty():
            print("No customers to serve")
            return
        customer = self.queue.dequeue()
        print(f"Serving customer {customer}")
        time.sleep(random.uniform(0.5, 2))  # Simulate service time
    
    def simulate(self, duration):
        for minute in range(duration):
            print(f"\nMinute {minute + 1}:")
            # Random chance of new customer arriving
            if random.random() < 0.6:  # 60% chance
                self.new_customer()
            # Serve a customer if queue not empty
            if not self.queue.is_empty():
                self.serve_customer()
            else:
                print("Counter idle")

# Run simulation
counter = TicketCounter()
counter.simulate(5)  # 5-minute simulation
```