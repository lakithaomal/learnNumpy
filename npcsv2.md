
# NumPy Cheat Sheet

NumPy (Numerical Python) is a fast, powerful library for numerical computing with fixed-type multidimensional arrays. It enables high-performance operations and forms the core of tools like Pandas and Matplotlib.

## Installation
```python
pip install numpy
```

## Import
```python
import numpy as np
```

## Array Creation
In numpy an array is called an ndarray `type(arr)` => `<class 'numpy.ndarray'>`
```python
np.array([1, 2, 3])                         # 1D array
np.array([[1, 2], [3, 4]])                  # 2D array
np.zeros(64)                               # 1D array of 64 zeros
np.zeros((2, 3))                           # 2x3 zero matrix
np.ones((2, 3))                            # 2x3 ones matrix
np.empty((2, 3))                           # 2x3 uninitialized array
np.arange(0, 10, 2)                        # Even numbers from 0 to 8
np.linspace(0, 1, 5)                       # 5 equally spaced values from 0 to 1 (inclusive)
np.identity(3)                             # 3x3 identity matrix
np.random.random((3, 3))                   # 3x3 matrix with random floats in [0, 1)
np.pad(np.ones((2, 2)), 1, constant_values=0)  # Pad 2x2 ones matrix with zeros
```

## Array Properties
```python
a = np.array([1, 2, 3, 4], dtype='int16')
a.shape            # (4,) - 1D array with 4 elements
a.ndim             # 1 - Number of dimensions
a.size             # 4 - Total number of elements
a.dtype            # int16 - Data type of elements
a.itemsize         # 2 - Bytes per element
```

## Reshape & Transpose
```python
a.reshape(2, 2)     # Reshape 1D array to 2x2
b = a.reshape(-1)   # You're flattening the array into 1D, because you're asking for a single dimension with all elements.
a.reshape(1, -1)	# 2D row vector (shape (1, n))
a.reshape(-1, 1)	# 2D column vector (shape (n,1))
a.T                 # Transpose of array (works for 2D+)
```

## Indexing & Slicing
```python
a[1, 2]             # Element at row 1, col 2 (2D array)
a[:, 1]             # All rows, column 1
a[1, :]             # Row 1, all columns
a[1:3, 0:2]         # Subarray (rows 1-2, cols 0-1)
a[::-1]             # Reverse array
np.where(a != 0)    # Indices of non-zero elements
q10.nonzero()       # Equivalent to np.where(q10 != 0)
```

## Arithmetic Operations
```python
a + b               # Element-wise addition
a - b               # Element-wise subtraction
a * b               # Element-wise multiplication
a / b               # Element-wise division
a // b              # Element-wise floor division
a % b               # Element-wise modulo
a ** 2              # Element-wise power
np.dot(a, b)        # Dot product
np.matmul(a, b)     # Matrix multiplication
```

## Statistical Operations
```python
a.min()             # Minimum value
a.max()             # Maximum value
a.sum()             # Sum of elements
a.mean()            # Mean of elements
a.std()             # Standard deviation
np.add.reduce(a)    # Sum via reduce
np.max(a, axis=1)   # Row-wise max
np.argmax(a)        # Index of max value
np.argmin(a)        # Index of min value
```

## Logical & Comparison
```python
a > 5                         # Element-wise comparison
np.any(a > 5)                 # Any element > 5?
np.all(a > 5)                 # All elements > 5?
np.where(a > 0, 1, 0)         # Conditional selection
np.intersect1d(a, b)          # Common elements (1D output)
np.union1d(a, b)              # All unique elements (1D output)
np.allclose(a, b)             # Approximate equality (with tolerance)
np.array_equal(a, b)          # Exact equality

```

## Bitwise & Special Operations
```python
2 << a >> 2         # Bitwise shift operations
1j * a              # Multiply by imaginary unit (complex numbers)
a // 2              # Floor division
a % 2               # Modulo
np.sqrt(4)          # Square root
np.emath.sqrt(-1)   # Complex sqrt (returns 1j); np.sqrt(-1) returns nan

```

## Tiling & Repeating
```python
np.tile([1,2], 3)                   # Repeat array: [1 2 1 2 1 2]
np.repeat([1,2], 2)                 # Repeat elements: [1 1 2 2]
np.tile([[1,2],[3,4]], (2,3))       # Tile 2D array 2x vertically and 3x horizontally
```

## Array Borders
```python
Z = np.ones((5, 5))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)

Z[:, [0, -1]] = 0   # Set first/last columns to 0
Z[[0, -1], :] = 0   # Set first/last rows to 0
```

## Normalization
```python
(a - np.mean(a)) / np.std(a)    # Normalize array (zero mean, unit std)
```

## Data Type Conversion
```python
a.astype(np.uint8)             # Convert data type
```

## Custom Data Type
```python
color = np.dtype([('r',np.ubyte),('g',np.ubyte),('b',np.ubyte),('a',np.ubyte)])
```

## Dates with NumPy
```python
np.datetime64('today')                             # Today's date
np.arange('2020-01', '2020-02', dtype='datetime64[D]')  # All Jan 2020 dates
```

## Mesh Grid
```python
x, y = np.meshgrid(np.linspace(0,1,3), np.linspace(0,1,3))
```

## Checkerboard Pattern
```python
Z = np.zeros((8,8), dtype=int)
Z[1::2, ::2] = 1
Z[::2, 1::2] = 1
```

## Matrix Operations
```python
np.linalg.det(a)        # Determinant
np.linalg.inv(a)        # Inverse
np.linalg.eig(a)        # Eigenvalues and eigenvectors
```

## Fancy Indexing
```python
Z[:, [0, -1]] = 0       # Set first/last columns to 0
Z[[0, -1], :] = 0       # Set first/last rows to 0
```

## Diagonal Matrix
```python
np.diag(1+np.arange(4), k=-1)   # Below main diagonal
```

## Unravel Index
```python
np.unravel_index(99, (6,7,8))   # Convert flat index to multi-dimensional index
```

## Linearly Spaced Vector
```python
np.linspace(0,1,12)     # 12 values from 0 to 1
```

## Sorting
```python
np.sort(a)              # Return sorted array
a.sort()                # In-place sort
```

## Stacking Arrays
```python
np.vstack([a,b])        # Vertical stack
np.hstack([a,b])        # Horizontal stack
```

## Loop Examples
```python
for item in iterable:
    print(item)

# Indexes only
for i in range(len(iterable)):
    print(i, iterable[i])

# Index + item
for i, item in enumerate(iterable):
    print(i, item)

# 2D index + value (for arrays)
for (i, j), val in np.ndenumerate(array):
    print((i, j), val)

# Something simple 
arr = np.array([[1, 2, 3], [4, 5, 6]])
for x in arr:
    print(x)
# [1 2 3]
# [4 5 6]

# N-dimensional Iterator - Gives you all the elements no matter the shape
for x in np.nditer(arr):
    print(x)
# 1
# 2
# 3
# 4
# 5
# 6

```

## Casting 
``` python
np.array([-1, 0, 1]).astype(bool) # [ True False  True] anything but a 0 is true even nans  
arr = np.array([1, 2, 3], dtype=str) # Initiating as string 
```

# NumPy: View vs Copy vs Assignment (`=`)

## 1. Using `=` : Assignment (No Copy)
When you assign an array using `=`, both variables point to the **same data** in memory.

### Example:
```python
import numpy as np

a = np.array([1, 2, 3])
b = a  # No copy, just a new reference
b[0] = 99

print(a)  # [99  2  3]  <-- a is modified too!
```

### Key Point:
- Changes in `b` affect `a`.
- `a` and `b` are the **same object** (`id(a) == id(b)`).

---

## 2. Using `copy()` : Deep Copy
Creates a **new array with its own data**, independent of the original.

### Example:
```python
a = np.array([1, 2, 3])
b = a.copy()  # Deep copy
b[0] = 99

print(a)  # [1 2 3]  <-- a is unchanged
print(b)  # [99 2 3]
```

### Key Point:
- `a` and `b` are **different objects**.
- Changes in `b` do **not affect** `a`.

---

## 3. Using `view()` : Shallow Copy (View)
Creates a **new array object** that shares the **same data** with the original. Changing the **data** affects both, but changing **shape** does not.

### Example:
```python
a = np.array([1, 2, 3])
b = a.view()
b[0] = 99

print(a)  # [99 2 3]  <-- data is shared
print(b)  # [99 2 3]
```

### Shape Change:
```python
a.shape = (3, 1)
print(a.shape)  # (3, 1)
print(b.shape)  # (3,)  <-- shape of b remains same
```

### Key Point:
- Different objects, **shared data**.
- Data change in one affects the other.
- Shape change in one does **not** affect the other.

---

## Summary Table:

| Method        | New Object? | Shared Data? | Changes Affect Original? |
|---------------|-------------|--------------|---------------------------|
| `=`           | No          | Yes          | Yes                       |
| `copy()`      | Yes         | No           | No                        |
| `view()`      | Yes         | Yes          | Yes (data), No (shape)    |
