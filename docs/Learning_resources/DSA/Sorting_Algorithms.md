# Sorting Algorithms in Python

## Introduction to Sorting

Sorting is the process of arranging data in a particular order (ascending or descending). It's one of the most fundamental operations in computer science with applications in databases, search algorithms, and data analysis.

### Types of Sorting:

1. **Internal Sorting**:
    - All data fits in main memory (RAM)
    - Faster access to elements
    - Examples: Bubble sort, Quick sort, Merge sort

2. **External Sorting**:
    - Data is too large to fit in main memory
    - Uses external storage (disk)
    - Examples: External merge sort, Polyphase merge sort

```python
# Internal vs External Sorting Example
def internal_sort_example(data):
    # All data in memory
    return sorted(data)

# Simulating external sort (using files)
def external_sort_example(input_file, output_file, chunk_size=1000):
    # Divide into sorted chunks
    chunks = []
    with open(input_file) as f:
        while True:
            chunk = []
            for _ in range(chunk_size):
                line = f.readline()
                if not line:
                    break
                chunk.append(int(line))
            if not chunk:
                break
            chunk.sort()
            chunks.append(chunk)
    
    # Merge chunks
    with open(output_file, 'w') as f:
        for num in sorted(sum(chunks, [])):
            f.write(f"{num}\n")
```

## Comparison Sorting Algorithms

### 1. Bubble Sort
Repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order.

**Time Complexity**: O(n²) worst and average case, O(n) best case (already sorted)

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        # Last i elements are already in place
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        # If no swaps, array is sorted
        if not swapped:
            break
    return arr

# Example
nums = [64, 34, 25, 12, 22, 11, 90]
print("Bubble Sort:", bubble_sort(nums.copy()))
```

### 2. Selection Sort
Divides the input list into a sorted and unsorted region, repeatedly selecting the smallest element from the unsorted region.

**Time Complexity**: O(n²) in all cases

```python
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# Example
print("Selection Sort:", selection_sort(nums.copy()))
```

### 3. Insertion Sort
Builds the final sorted array one item at a time by inserting each new item into its proper position.

**Time Complexity**: O(n²) worst and average case, O(n) best case

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr

# Example
print("Insertion Sort:", insertion_sort(nums.copy()))
```

### 4. Shell Sort
An optimization of insertion sort that allows exchange of items that are far apart by using gaps.

**Time Complexity**: O(n^(3/2)) or better depending on gap sequence

```python
def shell_sort(arr):
    n = len(arr)
    gap = n//2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j-gap] > temp:
                arr[j] = arr[j-gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr

# Example
print("Shell Sort:", shell_sort(nums.copy()))
```

## Divide and Conquer Sorting

### 1. Merge Sort
Divides the array into halves, sorts each half, then merges them back together.

**Time Complexity**: O(n log n) in all cases

```python
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]
        
        merge_sort(L)
        merge_sort(R)
        
        i = j = k = 0
        
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
    return arr

# Example
print("Merge Sort:", merge_sort(nums.copy()))
```

### 2. Quick Sort
Selects a 'pivot' element and partitions the array around the pivot.

**Time Complexity**: O(n log n) average case, O(n²) worst case

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# In-place version
def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return i+1

def quick_sort_inplace(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pi = partition(arr, low, high)
        quick_sort_inplace(arr, low, pi-1)
        quick_sort_inplace(arr, pi+1, high)
    return arr

# Example
print("Quick Sort:", quick_sort(nums.copy()))
print("Quick Sort (In-place):", quick_sort_inplace(nums.copy()))
```

### 3. Heap Sort
Converts the array into a max heap, then repeatedly extracts the maximum element.

**Time Complexity**: O(n log n) in all cases

```python
def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    
    if l < n and arr[i] < arr[l]:
        largest = l
    
    if r < n and arr[largest] < arr[r]:
        largest = r
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    
    # Build max heap
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements one by one
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

# Example
print("Heap Sort:", heap_sort(nums.copy()))
```

## Efficiency of Sorting Algorithms

### Time Complexity Comparison

| Algorithm      | Best Case | Average Case | Worst Case | Space Complexity | Stable |
|---------------|----------|--------------|------------|------------------|--------|
| Bubble Sort   | O(n)     | O(n²)        | O(n²)      | O(1)             | Yes    |
| Selection Sort | O(n²)    | O(n²)        | O(n²)      | O(1)             | No     |
| Insertion Sort | O(n)     | O(n²)        | O(n²)      | O(1)             | Yes    |
| Shell Sort    | O(n log n)| O(n^(3/2))   | O(n²)      | O(1)             | No     |
| Merge Sort    | O(n log n)| O(n log n)   | O(n log n) | O(n)             | Yes    |
| Quick Sort    | O(n log n)| O(n log n)   | O(n²)      | O(log n)          | No     |
| Heap Sort     | O(n log n)| O(n log n)   | O(n log n) | O(1)             | No     |

### When to Use Which Algorithm:

1. **Small datasets (n < 100)**:
    - Insertion sort (simple, low overhead)
   
2. **Medium datasets (100 < n < 10,000)**:
    - Shell sort or Quick sort (good performance)

3. **Large datasets (n > 10,000)**:
    - Merge sort or Heap sort (guaranteed O(n log n))
    - Quick sort (generally fastest in practice)

4. **Special Cases**:
    - Almost sorted data: Insertion sort (approaches O(n))
    - Stability required: Merge sort or Insertion sort
    - Memory constrained: Heap sort (O(1) space)

```python
# Performance comparison function
import time
import random

def test_sort(sort_func, data):
    start = time.time()
    sort_func(data.copy())
    end = time.time()
    return end - start

# Generate test data
small_data = random.sample(range(100), 50)
medium_data = random.sample(range(10000), 1000)
large_data = random.sample(range(1000000), 100000)

# Test all algorithms
algorithms = {
    "Bubble Sort": bubble_sort,
    "Selection Sort": selection_sort,
    "Insertion Sort": insertion_sort,
    "Shell Sort": shell_sort,
    "Merge Sort": merge_sort,
    "Quick Sort": quick_sort,
    "Heap Sort": heap_sort
}

print("Small Data (50 elements):")
for name, func in algorithms.items():
    print(f"{name}: {test_sort(func, small_data):.6f} seconds")

print("\nMedium Data (1000 elements):")
for name, func in algorithms.items():
    if name not in ["Bubble Sort", "Selection Sort"]:  # Too slow
        print(f"{name}: {test_sort(func, medium_data):.6f} seconds")

print("\nLarge Data (100000 elements):")
for name, func in algorithms.items():
    if name in ["Merge Sort", "Quick Sort", "Heap Sort"]:
        print(f"{name}: {test_sort(func, large_data):.6f} seconds")
```

### Key Takeaways:
1. **O(n²) algorithms** are only suitable for small datasets
2. **Quick sort** is generally the fastest in practice
3. **Merge sort** is stable and has consistent performance
4. **Heap sort** is good when memory is limited
5. The best algorithm depends on your specific data characteristics and requirements