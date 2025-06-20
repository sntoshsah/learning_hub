# Greedy Algorithms: Concepts and Applications

## Introduction to Greedy Algorithms

Greedy algorithms are a class of algorithms that make locally optimal choices at each step with the hope of finding a global optimum. They are simple, intuitive, and often efficient, but don't always produce the optimal solution for all problems.

### Key Characteristics:
1. **Greedy Choice Property**: A global optimum can be reached by making locally optimal choices
2. **Optimal Substructure**: An optimal solution contains optimal solutions to subproblems
3. **Irrevocable**: Choices made cannot be changed later

## When to Use Greedy Algorithms

Greedy algorithms work well for problems where:

- Local optimal choices lead to global optimum
- The problem has optimal substructure
- We can make a decision without considering future choices
- The solution can be built incrementally

## Classic Greedy Problems with Python Implementations

### 1. Activity Selection Problem

**Problem**: Select the maximum number of non-overlapping activities from a set.

#### Greedy Approach:
1. Sort activities by finish time
2. Select first activity
3. For each remaining activity, select if it doesn't conflict with last selected

```python
def activity_selection(start, finish):
    n = len(finish)
    selected = []
    
    # Sort activities by finish time
    activities = sorted(zip(start, finish), key=lambda x: x[1])
    
    # Select first activity
    i = 0
    selected.append(i)
    
    for j in range(1, n):
        # If activity starts after last finishes, select it
        if activities[j][0] >= activities[i][1]:
            selected.append(j)
            i = j
    
    return selected

# Example
start = [1, 3, 0, 5, 8, 5]
finish = [2, 4, 6, 7, 9, 9]
selected = activity_selection(start, finish)
print("Selected activities:", selected)  # Output: [0, 1, 3, 4]
```

### 2. Fractional Knapsack Problem

**Problem**: Fill a knapsack with maximum value, allowing fractions of items.

#### Greedy Approach:
1. Calculate value/weight ratio for each item
2. Sort items by ratio in descending order
3. Take as much as possible of highest ratio items

```python
def fractional_knapsack(value, weight, capacity):
    n = len(value)
    items = list(zip(value, weight))
    
    # Sort by value/weight ratio (descending)
    items.sort(key=lambda x: x[0]/x[1], reverse=True)
    
    total_value = 0.0
    
    for v, w in items:
        if capacity >= w:
            total_value += v
            capacity -= w
        else:
            fraction = capacity / w
            total_value += v * fraction
            break
    
    return total_value

# Example
value = [60, 100, 120]
weight = [10, 20, 30]
capacity = 50
print("Maximum value:", fractional_knapsack(value, weight, capacity))
# Output: 240.0 (all of items 1 and 2, 2/3 of item 3)
```

### 3. Huffman Coding (Data Compression)

**Problem**: Assign variable-length codes to characters to minimize total bits.

#### Greedy Approach:
1. Build frequency table
2. Create min-heap of nodes
3. Repeatedly combine two smallest nodes
4. Assign codes based on tree paths

```python
import heapq

class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    if not text:
        return None
    
    # Calculate frequency
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    
    # Create priority queue
    heap = []
    for char, count in freq.items():
        heapq.heappush(heap, HuffmanNode(char, count))
    
    # Build tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    return heapq.heappop(heap)

def build_codes(node, prefix="", codebook={}):
    if node is None:
        return
    
    if node.char is not None:
        codebook[node.char] = prefix
    
    build_codes(node.left, prefix + "0", codebook)
    build_codes(node.right, prefix + "1", codebook)
    
    return codebook

# Example
text = "this is an example for huffman encoding"
huffman_tree = build_huffman_tree(text)
codes = build_codes(huffman_tree)
print("Huffman Codes:", codes)
```

### 4. Dijkstra's Algorithm (Shortest Path)

**Problem**: Find shortest paths from a source to all other vertices.

#### Greedy Approach:
1. Maintain set of unvisited nodes
2. At each step, pick node with smallest distance
3. Update distances to its neighbors
4. Repeat until all nodes visited

```python
import heapq

def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    heap = [(0, start)]
    
    while heap:
        current_dist, current_vertex = heapq.heappop(heap)
        
        if current_dist > distances[current_vertex]:
            continue
        
        for neighbor, weight in graph[current_vertex].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
    
    return distances

# Example
graph = {
    'A': {'B': 2, 'C': 5},
    'B': {'A': 2, 'C': 1, 'D': 7},
    'C': {'A': 5, 'B': 1, 'D': 3},
    'D': {'B': 7, 'C': 3}
}
print("Shortest distances:", dijkstra(graph, 'A'))
# Output: {'A': 0, 'B': 2, 'C': 3, 'D': 6}
```

### 5. Minimum Spanning Tree (Prim's Algorithm)

**Problem**: Find a minimum weight spanning tree for a connected weighted graph.

#### Greedy Approach:
1. Start with arbitrary vertex
2. At each step, add cheapest edge that connects tree to new vertex
3. Repeat until all vertices are included

```python
import heapq

def prim_mst(graph):
    mst = []
    visited = set()
    start_vertex = next(iter(graph))
    heap = []
    
    visited.add(start_vertex)
    for neighbor, weight in graph[start_vertex].items():
        heapq.heappush(heap, (weight, start_vertex, neighbor))
    
    while heap and len(visited) < len(graph):
        weight, u, v = heapq.heappop(heap)
        if v not in visited:
            visited.add(v)
            mst.append((u, v, weight))
            for neighbor, w in graph[v].items():
                if neighbor not in visited:
                    heapq.heappush(heap, (w, v, neighbor))
    
    return mst

# Example
graph = {
    'A': {'B': 2, 'D': 6},
    'B': {'A': 2, 'C': 3, 'D': 8},
    'C': {'B': 3, 'D': 4},
    'D': {'A': 6, 'B': 8, 'C': 4}
}
print("MST edges:", prim_mst(graph))
# Output: [('A', 'B', 2), ('B', 'C', 3), ('C', 'D', 4)]
```

### 6. Coin Change Problem (Greedy Version)

**Problem**: Make change using fewest coins (when greedy works).

#### Greedy Approach:
1. Sort coins in descending order
2. Use as many as possible of largest coin
3. Move to next smaller coin

```python
def coin_change_greedy(coins, amount):
    coins.sort(reverse=True)
    change = []
    
    for coin in coins:
        while amount >= coin:
            amount -= coin
            change.append(coin)
    
    return change if amount == 0 else None

# Example (works for standard coin systems)
coins = [25, 10, 5, 1]
amount = 63
print("Change for 63 cents:", coin_change_greedy(coins, amount))
# Output: [25, 25, 10, 1, 1, 1]
```

### 7. Job Sequencing with Deadlines

**Problem**: Schedule jobs to maximize profit before deadlines.

#### Greedy Approach:
1. Sort jobs by profit (descending)
2. Schedule each job as late as possible before deadline

```python
def job_sequencing(jobs):
    # Sort jobs by profit (descending)
    jobs.sort(key=lambda x: x[2], reverse=True)
    
    max_deadline = max(job[1] for job in jobs)
    schedule = [None] * (max_deadline + 1)
    
    for job in jobs:
        id, deadline, profit = job
        # Find latest available slot before deadline
        for i in range(deadline, 0, -1):
            if schedule[i] is None:
                schedule[i] = id
                break
    
    # Remove None values and return scheduled jobs
    return [job for job in schedule if job is not None]

# Example
jobs = [
    ('a', 2, 100),
    ('b', 1, 19),
    ('c', 2, 27),
    ('d', 1, 25),
    ('e', 3, 15)
]
print("Scheduled jobs:", job_sequencing(jobs))
# Output: ['c', 'a', 'e']
```

## Greedy vs Dynamic Programming

| Characteristic | Greedy | Dynamic Programming |
|---------------|--------|---------------------|
| **Approach** | Makes locally optimal choice | Considers all possibilities |
| **Optimality** | Not always optimal | Always optimal |
| **Speed** | Faster | Slower |
| **Space** | Usually less memory | More memory |
| **Examples** | Dijkstra's, Prim's | Knapsack, LCS |

## When Greedy Fails

Greedy algorithms don't always produce optimal solutions. Example: The classic 0/1 Knapsack problem cannot be solved optimally with a greedy approach (except in fractional version).

```python
# Non-optimal greedy solution for 0/1 Knapsack
def greedy_knapsack(value, weight, capacity):
    ratio = [v/w for v, w in zip(value, weight)]
    indices = sorted(range(len(ratio)), key=lambda i: ratio[i], reverse=True)
    
    total_value = 0
    total_weight = 0
    
    for i in indices:
        if total_weight + weight[i] <= capacity:
            total_value += value[i]
            total_weight += weight[i]
    
    return total_value

# Example where greedy fails
value = [60, 100, 120]
weight = [10, 20, 30]
capacity = 50
print("Greedy knapsack:", greedy_knapsack(value, weight, capacity))  # 160
print("Optimal solution:", 220)  # Using DP would give 220
```

## Proof Techniques for Greedy Algorithms

To verify a greedy algorithm's correctness:

1. **Greedy Choice Property**: Prove that greedy choice is part of some optimal solution
2. **Optimal Substructure**: Show that remaining subproblem is similar to original
3. **Exchange Argument**: Show that any optimal solution can be transformed to include greedy choice

Greedy algorithms are powerful tools when applicable, offering efficient solutions to many optimization problems. The key is recognizing when the greedy choice property holds and when it doesn't.