# Trees and Graphs in Python

## Concept and Definitions

### Tree Basics
A tree is a hierarchical data structure consisting of nodes connected by edges with these properties:
- One root node
- Each node has zero or more child nodes
- No cycles (a node can't be its own ancestor)

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# Creating a simple tree
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
```

### Key Terminology:
- **Root**: Topmost node
- **Parent/Child**: Direct connections between nodes
- **Leaf**: Node with no children
- **Depth**: Number of edges from root to node
- **Height**: Number of edges on longest path from node to leaf
- **Level**: Depth + 1 (root is level 1)

## Basic Operations in Binary Tree

### Tree Height Calculation
```python
def tree_height(node):
    if node is None:
        return -1  # or 0 if counting nodes instead of edges
    return max(tree_height(node.left), tree_height(node.right)) + 1

print("Tree height:", tree_height(root))  # Output: 2
```

### Node Level and Depth
```python
def node_depth(node, target, depth=0):
    if node is None:
        return -1
    if node.value == target:
        return depth
    left = node_depth(node.left, target, depth+1)
    if left != -1:
        return left
    return node_depth(node.right, target, depth+1)

print("Depth of node 5:", node_depth(root, 5))  # Output: 2
```

## Binary Search Tree (BST)

A BST is a binary tree where for each node:
- All left descendants ≤ node value
- All right descendants > node value

### BST Insertion
```python
def bst_insert(root, value):
    if root is None:
        return TreeNode(value)
    if value <= root.value:
        root.left = bst_insert(root.left, value)
    else:
        root.right = bst_insert(root.right, value)
    return root

# Building a BST
bst_root = None
for num in [5, 3, 7, 2, 4, 6, 8]:
    bst_root = bst_insert(bst_root, num)
```

### BST Search
```python
def bst_search(root, target):
    if root is None:
        return False
    if root.value == target:
        return True
    if target < root.value:
        return bst_search(root.left, target)
    return bst_search(root.right, target)

print("Is 6 in BST?", bst_search(bst_root, 6))  # True
print("Is 9 in BST?", bst_search(bst_root, 9))  # False
```

### BST Deletion
```python
def bst_delete(root, key):
    if root is None:
        return root
    
    if key < root.value:
        root.left = bst_delete(root.left, key)
    elif key > root.value:
        root.right = bst_delete(root.right, key)
    else:
        # Node with only one child or no child
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left
        
        # Node with two children
        root.value = min_value(root.right)
        root.right = bst_delete(root.right, root.value)
    
    return root

def min_value(node):
    current = node
    while current.left:
        current = current.left
    return current.value

# Delete node 3
bst_root = bst_delete(bst_root, 3)
```

### Tree Traversals
```python
def inorder_traversal(node):
    if node:
        inorder_traversal(node.left)
        print(node.value, end=" ")
        inorder_traversal(node.right)

def preorder_traversal(node):
    if node:
        print(node.value, end=" ")
        preorder_traversal(node.left)
        preorder_traversal(node.right)

def postorder_traversal(node):
    if node:
        postorder_traversal(node.left)
        postorder_traversal(node.right)
        print(node.value, end=" ")

print("Inorder:", end=" ")
inorder_traversal(bst_root)  # 2 4 5 6 7 8

print("\nPreorder:", end=" ")
preorder_traversal(bst_root)  # 5 4 2 7 6 8

print("\nPostorder:", end=" ")
postorder_traversal(bst_root)  # 2 4 6 8 7 5
```

## AVL Tree (Balanced BST)

AVL trees maintain balance with rotations when the height difference between left and right subtrees (balance factor) exceeds 1.

### AVL Implementation
```python
class AVLNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

def get_height(node):
    if not node:
        return 0
    return node.height

def get_balance(node):
    if not node:
        return 0
    return get_height(node.left) - get_height(node.right)

def left_rotate(z):
    y = z.right
    T2 = y.left
    
    y.left = z
    z.right = T2
    
    z.height = 1 + max(get_height(z.left), get_height(z.right))
    y.height = 1 + max(get_height(y.left), get_height(y.right))
    
    return y

def right_rotate(z):
    y = z.left
    T3 = y.right
    
    y.right = z
    z.left = T3
    
    z.height = 1 + max(get_height(z.left), get_height(z.right))
    y.height = 1 + max(get_height(y.left), get_height(y.right))
    
    return y

def avl_insert(root, value):
    if not root:
        return AVLNode(value)
    
    if value < root.value:
        root.left = avl_insert(root.left, value)
    else:
        root.right = avl_insert(root.right, value)
    
    root.height = 1 + max(get_height(root.left), get_height(root.right))
    
    balance = get_balance(root)
    
    # Left Left Case
    if balance > 1 and value < root.left.value:
        return right_rotate(root)
    
    # Right Right Case
    if balance < -1 and value > root.right.value:
        return left_rotate(root)
    
    # Left Right Case
    if balance > 1 and value > root.left.value:
        root.left = left_rotate(root.left)
        return right_rotate(root)
    
    # Right Left Case
    if balance < -1 and value < root.right.value:
        root.right = right_rotate(root.right)
        return left_rotate(root)
    
    return root

avl_root = None
for num in [10, 20, 30, 40, 50, 25]:
    avl_root = avl_insert(avl_root, num)
```

## Applications of Trees

1. **File Systems**: Directory structure
2. **Database Indexing**: B-trees, B+ trees
3. **Networking**: Routing tables
4. **Compression**: Huffman coding trees
5. **AI**: Decision trees
6. **XML/HTML Parsing**: DOM trees

## Graph Concepts

### Graph Definitions
A graph G = (V, E) consists of:
- V: Set of vertices (nodes)
- E: Set of edges (connections between nodes)

### Graph Representations
1. **Adjacency Matrix**:
```python
# Undirected graph
graph_matrix = [
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [0, 1, 0, 0]
]
```

2. **Adjacency List**:
```python
graph_list = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B'],
    'D': ['B']
}
```

### Graph Traversals

#### Breadth-First Search (BFS)
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        vertex = queue.popleft()
        print(vertex, end=" ")
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

print("BFS:")
bfs(graph_list, 'A')  # A B C D
```

#### Depth-First Search (DFS)
```python
def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    print(node, end=" ")
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

print("\nDFS:")
dfs(graph_list, 'A')  # A B C D
```

## Minimum Spanning Trees

### Kruskal's Algorithm
```python
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []
    
    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])
    
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])
    
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
    
    def kruskal_mst(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i += 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            
            if x != y:
                e += 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
        
        print("Kruskal's MST:")
        for u, v, weight in result:
            print(f"{u} -- {v} == {weight}")

g = Graph(4)
g.add_edge(0, 1, 10)
g.add_edge(0, 2, 6)
g.add_edge(0, 3, 5)
g.add_edge(1, 3, 15)
g.add_edge(2, 3, 4)
g.kruskal_mst()
```

### Prim's Algorithm
```python
import sys

class PrimGraph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]
    
    def prim_mst(self):
        key = [sys.maxsize] * self.V
        parent = [None] * self.V
        key[0] = 0
        mst_set = [False] * self.V
        
        parent[0] = -1
        
        for _ in range(self.V):
            u = self.min_key(key, mst_set)
            mst_set[u] = True
            
            for v in range(self.V):
                if self.graph[u][v] > 0 and not mst_set[v] and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u
        
        self.print_mst(parent)
    
    def min_key(self, key, mst_set):
        min_val = sys.maxsize
        min_index = -1
        
        for v in range(self.V):
            if key[v] < min_val and not mst_set[v]:
                min_val = key[v]
                min_index = v
        return min_index
    
    def print_mst(self, parent):
        print("Prim's MST:")
        for i in range(1, self.V):
            print(f"{parent[i]} -- {i} == {self.graph[i][parent[i]]}")

g = PrimGraph(5)
g.graph = [
    [0, 2, 0, 6, 0],
    [2, 0, 3, 8, 5],
    [0, 3, 0, 0, 7],
    [6, 8, 0, 0, 9],
    [0, 5, 7, 9, 0]
]
g.prim_mst()
```

## Shortest Path Algorithms

### Dijkstra's Algorithm
```python
import heapq

def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_dist, current_vertex = heapq.heappop(pq)
        
        if current_dist > distances[current_vertex]:
            continue
        
        for neighbor, weight in graph[current_vertex].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

graph = {
    'A': {'B': 2, 'C': 6},
    'B': {'D': 5},
    'C': {'D': 8},
    'D': {}
}

print("Dijkstra's shortest paths:")
print(dijkstra(graph, 'A'))  # {'A': 0, 'B': 2, 'C': 6, 'D': 7}
```