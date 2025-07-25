# Searching and Hashing in Python

## Introduction to Searching

Searching is the process of finding a particular element in a collection of data. Efficient searching is crucial for performance in many applications like databases, compilers, and AI systems.

### Types of Searching:
1. **Internal Searching**: Data fits in main memory (RAM)
2. **External Searching**: Data is stored in external storage (disk)

### Key Concepts:
- **Search Key**: The value being searched for
- **Search Space**: The collection of data being searched
- **Success/Failure**: Whether the key is found or not

```python
# Basic search example
def simple_search(items, target):
    """Returns True if target is found in items"""
    for item in items:
        if item == target:
            return True
    return False

numbers = [4, 2, 7, 1, 9, 5]
print(simple_search(numbers, 7))  # True
print(simple_search(numbers, 3))  # False
```

## Search Algorithms

### 1. Sequential (Linear) Search
Checks each element in order until the target is found.

**Time Complexity**:

- Best case: O(1) (first element)
- Average case: O(n)
- Worst case: O(n) (last element or not present)

```python
def linear_search(arr, target):
    """Returns index of target if found, -1 otherwise"""
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Example
arr = [10, 20, 30, 40, 50]
print(linear_search(arr, 30))  # 2
print(linear_search(arr, 35))  # -1
```

### 2. Binary Search
Efficiently searches a sorted array by repeatedly dividing the search interval in half.

**Time Complexity**:

- Best case: O(1) (middle element)
- Average case: O(log n)
- Worst case: O(log n)

```python
def binary_search(arr, target):
    """Returns index of target in sorted arr, -1 if not found"""
    low = 0
    high = len(arr) - 1
    
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Example
sorted_arr = [10, 20, 30, 40, 50, 60]
print(binary_search(sorted_arr, 40))  # 3
print(binary_search(sorted_arr, 45))  # -1

# Recursive version
def binary_search_recursive(arr, target, low, high):
    if low > high:
        return -1
    mid = (low + high) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid+1, high)
    else:
        return binary_search_recursive(arr, target, low, mid-1)

# Wrapper function
def binary_search_rec(arr, target):
    return binary_search_recursive(arr, target, 0, len(arr)-1)

print(binary_search_rec(sorted_arr, 20))  # 1
```

## Efficiency of Search Algorithms

### Comparison Table:

| Algorithm        | Best Case | Average Case | Worst Case | Space Complexity | Requires Sorted Data |
|-----------------|----------|--------------|------------|------------------|----------------------|
| Sequential Search | O(1)     | O(n)         | O(n)       | O(1)             | No                   |
| Binary Search    | O(1)     | O(log n)     | O(log n)   | O(1)             | Yes                  |

### When to Use Which:

1. **Unsorted data or small datasets**:
    - Linear search (simpler implementation)

2. **Large sorted datasets**:
    - Binary search (much more efficient)

3. **Frequent searches on static data**:
    - Sort first then use binary search

```python
# Performance comparison
import time
import random

def test_search(search_func, data, target):
    start = time.perf_counter()
    result = search_func(data, target)
    end = time.perf_counter()
    return (result, end - start)

# Generate test data
small_data = random.sample(range(100), 50)
large_data = sorted(random.sample(range(1000000), 100000))
target_present = large_data[50000]
target_absent = -1

# Test searches
print("Small dataset (50 elements, unsorted):")
res, time_taken = test_search(linear_search, small_data, small_data[25])
print(f"Linear search: {time_taken:.8f} seconds")

print("\nLarge dataset (100,000 elements, sorted):")
res, time_taken = test_search(linear_search, large_data, target_present)
print(f"Linear search (present): {time_taken:.8f} seconds")

res, time_taken = test_search(binary_search, large_data, target_present)
print(f"Binary search (present): {time_taken:.8f} seconds")

res, time_taken = test_search(linear_search, large_data, target_absent)
print(f"Linear search (absent): {time_taken:.8f} seconds")

res, time_taken = test_search(binary_search, large_data, target_absent)
print(f"Binary search (absent): {time_taken:.8f} seconds")
```

## Hashing: Hash Functions and Hash Tables

Hashing is a technique that maps data of arbitrary size to fixed-size values (hash codes) for efficient lookup.

### Key Components:
1. **Hash Function**: Maps keys to array indices
2. **Hash Table**: Data structure that stores key-value pairs
3. **Collision Resolution**: Handling when two keys hash to the same index

### Properties of Good Hash Functions:
1. **Deterministic**: Same input → same output
2. **Uniform Distribution**: Spreads keys evenly
3. **Efficient to Compute**: Fast calculation
4. **Minimal Collisions**: Few key overlaps

```python
# Simple hash function example
def simple_hash(key, size):
    """Basic hash function using modulo"""
    return sum(ord(c) for c in str(key)) % size

# Example usage
print(simple_hash("hello", 10))  # 2
print(simple_hash("world", 10))  # 9
print(simple_hash(12345, 10))    # 5

# Python's built-in hash table: dictionary
hash_table = {}
hash_table["apple"] = 1.00
hash_table["banana"] = 0.50
hash_table["orange"] = 0.75

print(hash_table.get("banana"))  # 0.5
print(hash_table.get("grape", "Not found"))  # Not found
```

## Collision Resolution Techniques

### 1. Separate Chaining
Each bucket contains a linked list of entries.

```python
class HashTableChaining:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def insert(self, key, value):
        hash_key = self._hash(key)
        bucket = self.table[hash_key]
        
        # Check if key already exists
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
    
    def search(self, key):
        hash_key = self._hash(key)
        bucket = self.table[hash_key]
        
        for k, v in bucket:
            if k == key:
                return v
        return None
    
    def delete(self, key):
        hash_key = self._hash(key)
        bucket = self.table[hash_key]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                return
        raise KeyError(key)

# Example usage
ht = HashTableChaining(10)
ht.insert("apple", 1.00)
ht.insert("banana", 0.50)
ht.insert("orange", 0.75)
ht.insert("apple", 1.20)  # Update

print(ht.search("banana"))  # 0.50
print(ht.search("apple"))   # 1.20
ht.delete("orange")
print(ht.search("orange"))  # None
```

### 2. Open Addressing
All entries are stored in the array itself. When collision occurs, it finds the next available slot.

#### Common Probing Methods:
- Linear Probing
- Quadratic Probing
- Double Hashing

```python
class HashTableOpenAddressing:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size
        self.deleted = object()  # Marker for deleted items
    
    def _hash(self, key, i=0):
        # Linear probing: h(k, i) = (h'(k) + i) mod m
        return (hash(key) + i) % self.size
    
    def insert(self, key, value):
        for i in range(self.size):
            hash_key = self._hash(key, i)
            if self.table[hash_key] is None or self.table[hash_key] is self.deleted:
                self.table[hash_key] = (key, value)
                return
        raise Exception("Hash table is full")
    
    def search(self, key):
        for i in range(self.size):
            hash_key = self._hash(key, i)
            entry = self.table[hash_key]
            
            if entry is None:
                return None
            if entry is not self.deleted and entry[0] == key:
                return entry[1]
        return None
    
    def delete(self, key):
        for i in range(self.size):
            hash_key = self._hash(key, i)
            entry = self.table[hash_key]
            
            if entry is None:
                raise KeyError(key)
            if entry is not self.deleted and entry[0] == key:
                self.table[hash_key] = self.deleted
                return
        raise KeyError(key)

# Example usage
ht_oa = HashTableOpenAddressing(10)
ht_oa.insert("apple", 1.00)
ht_oa.insert("banana", 0.50)
ht_oa.insert("orange", 0.75)

print(ht_oa.search("banana"))  # 0.50
ht_oa.delete("banana")
print(ht_oa.search("banana"))  # None
```

## Practical Hashing Applications

### 1. Counting Word Frequencies
```python
def word_frequency(text):
    freq = {}
    for word in text.split():
        freq[word] = freq.get(word, 0) + 1
    return freq

text = "apple banana orange apple apple orange"
print(word_frequency(text))
# {'apple': 3, 'banana': 1, 'orange': 2}
```

### 2. Implementing a Cache (LRU Cache)
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Example usage
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # 1
cache.put(3, 3)      # Evicts key 2
print(cache.get(2))  # -1 (not found)
```

### 3. Password Storage (with Salting)
```python
import hashlib
import os

def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(32)  # Random salt
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000  # Number of iterations
    )
    return salt + key

def verify_password(stored_hash, password):
    salt = stored_hash[:32]
    key = stored_hash[32:]
    new_hash = hash_password(password, salt)
    return key == new_hash[32:]

# Example usage
password = "securepassword123"
hashed = hash_password(password)
print(f"Stored hash: {hashed.hex()}")
print(verify_password(hashed, password))        # True
print(verify_password(hashed, "wrongpass"))    # False
```