
# NumPy Cheat Sheet

NumPy is a fast, powerful library for numerical computing with fixed-type multidimensional arrays, enabling high-performance operations and forming the core of tools like Pandas and Matplotlib.

---

## Installation
```python
pip install numpy
```

---

## Import
```python
import numpy as np
```

---

## Array Creation
```python
np.array([1, 2, 3])                         # 1D array
np.array([[1, 2], [3, 4]])                  # 2D array
np.zeros(64)                                # 1D matrix with 64 zeros  
np.zeros((2, 3))                            # 2x3 zero matrix
np.ones((2, 3))                             # 2x3 ones matrix
np.empty((2, 3))                            # 2x3 uninitialized
np.arange(0, 10, 2)                         # 0 to 8 step 2 - 10 is not inclusive 
np.linspace(0, 1, 5)                        # 5 values 0 to 1 - 0 and 1 are inclusive  
np.identity(3)                              # 3x3 identity matrix 
np.random.random((3, 3)) # similar to  np.random.rand(3, 3) 3x3 random matrix - Values are [0,1)
np.pad(np.ones((2,2)), 1, constant_values=0)# Padding with zeros
```

---

## Array Properties
```python
a = np.array([1,2,3,4], 'int16')
a.shape            #  gives (4,)                                 # (rows, cols)
a.ndim            # gives Dimensions 1                                # Number of dimensions
a.size                # gives 4                       # Total elements
a.dtype             # int64                        # Data type
a.itemsize         # gives 2                         # Bytes per element
```

---

## Reshape & Transpose
```python
a.reshape(2, 3)                             # Reshape array
a.T                                         # Transpose
```

---

## Indexing & Slicing
```python
a[1, 2]                                     # Element at row 1, col 2
a[:, 1]                                     # All rows, col 1
a[1, :]                                     # Row 1, all cols
a[1:3, 0:2]                                 # Subarray
a[::-1]                                     # Reverse array
np.where(a != 0)   or q10.nonzero()         # Non-zero indices
```

---

## Arithmetic Operations
```python
a + b                                       # Element-wise addition
a - b                                       # Subtraction
a * b                                       # Multiplication
a / b                                       # Division
a//b                                        # Floor Division
a%b                                         # Gives modulas 
a ** 2                                      # Power
np.dot(a, b)                                # Dot product
np.matmul(a, b)                             # Matrix multiplication
```

---

## Statistical Operations
```python
a.min()                                     # Minimum value
a.max()                                     # Maximum value
a.sum()                                     # Sum of elements
a.mean()                                    # Mean
a.std()                                     # Standard deviation
np.add.reduce(a)                           # Sum via reduce
np.max(a, axis=1)                           # Row-wise max
np.argmax(a)                                # Index of max
np.argmin(a)                                # Index of min
```

---

## Logical & Comparison
```python
a > 5                                       # Element-wise comparison
np.any(a > 5)                               # Any > 5?
np.all(a > 5)                               # All > 5?
np.where(a > 0, 1, 0)                       # Conditional select
np.intersect1d(a, b)                        # Common elements - always gives 1d output
np.union1d(a, b)                            # All unique elements - always gives 1d output
np.allclose(a, b)                           # Approximately equal - True or False outputs can add tolerance
np.array_equal(a, b)                        # Exactly equal - True or False outputs 
```

---

## Bitwise & Special
```python
2 << a >> 2                                 # Bit shifting
1j * a                                      # Complex multiplication
a // 2                                      # Floor division
a % 2                                       # Modulo
np.sqrt(4)                                  # Square root
np.emath.sqrt(-1)                           # Complex sqrt np.sqrt(-1) returns nan and np.emath.sqrt(-1)  returns 1j 
```

---

## Tiling & Repeating
```python
np.tile([1,2], 3)                           # Repeat array: [1 2 1 2 1 2]
np.repeat([1,2], 2)                         # Repeat elements: [1 1 2 2]

# 2D Tiling
np.tile([[1,2],[3,4]], (2,3))
# Output:
# [[1 2 1 2 1 2]
#  [3 4 3 4 3 4]
#  [1 2 1 2 1 2]
#  [3 4 3 4 3 4]]
# The array is repeated 2 times vertically (rows),
# And 3 times horizontally (columns).
```

## Array Border 
```python
Z = np.ones((5,5))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)

# Using fancy indexing
Z[:, [0, -1]] = 0
Z[[0, -1], :] = 0
```

---

## Normalization
```python
(a - np.mean(a)) / np.std(a)                # Normalize
```

---

## Data Type Conversion
```python
a.astype(np.uint8)                          # Cast type
```

---

## Custom Data Type
```python
color = np.dtype([('r',np.ubyte),('g',np.ubyte),('b',np.ubyte),('a',np.ubyte)])
```

---

## Dates with NumPy
```python
np.datetime64('today')                                  # Today's date
np.arange('2020-01', '2020-02', dtype='datetime64[D]')  # All Jan dates
```

---

## Mesh Grid
```python
x, y = np.meshgrid(np.linspace(0,1,3), np.linspace(0,1,3))
```

---

## Checkerboard Pattern
```python
Z = np.zeros((8,8), dtype=int); Z[1::2,::2]=1; Z[::2,1::2]=1
```

---

## Matrix Operations
```python
np.linalg.det(a)                            # Determinant
np.linalg.inv(a)                            # Inverse
np.linalg.eig(a)                            # Eigenvalues/vectors
```

---

## Fancy Indexing
```python
Z[:, [0, -1]] = 0                           # Set first/last col to 0
Z[[0, -1], :] = 0                           # Set first/last row to 0
```

---

## Diagonal Matrix
```python
np.diag(1+np.arange(4), k=-1)               # Below diagonal
```

---

## Unravel Index
```python
np.unravel_index(99, (6,7,8))               # Convert flat index to coords
```

---

## Linearly Spaced Vector
```python
np.linspace(0,1,12)                         # 12 values from 0 to 1
```

---

## Sorting
```python
np.sort(a)                                  # Return sorted array
a.sort()                                    # Sort in-place
```

---

## Stacking Arrays
```python
np.vstack([a,b])                            # Vertical stack
np.hstack([a,b])                            # Horizontal stack

v1 = np.array([1,2,3,4])
v2 = np.array([4,5,6,7])
v3 = np.vstack([v1,v2])
# [[1 2 3 4]
#  [4 5 6 7]]
v4 = np.hstack([v1,v2])
print(v4) # [1 2 3 4 4 5 6 7]
```

---

## Sum Differences
```python
sum(range(5), -1)                           # Python: 9
np.sum(range(5), -1)                        # NumPy: 10
```

---

## Range functions 
```python 
range(5)   # goes from 0 and does not include 5
range(4,8) # starts from 4 ends in 7
```


### f-Strings (Modern)
```python

apples = 5
price = 3.1415
name = "Alice"

print(f"I have {apples} apples")             # integer
print(f"Price: {price:.2f}")                 # float with 2 decimal places
print(f"Hello, {name}")                      # string
print(f"Hex: {255:x}")                       # hex
print(f"Sci: {12345.678:e}")                 # scientific notation                         # Hexadecimal
```
### Old Style
```python
"I have %d apples" % 5                      # Integer
"Price: %.2f" % 3.1415                      # Float 2 decimals
"Hex: %x" % 255                             # Hexadecimal
```
---

## Help & Info
```python
np.info('add')                              # Info on function
```
# 🔁 Python Loops – NumPy Style

| Task             | Best Practice               |
|------------------|-----------------------------|
| Items only       | `for item in iterable`      |
| Indexes only     | `for i in range(len(...))`  |
| Index + item     | `enumerate(iterable)`       |
| 2D index + val   | `np.ndenumerate(array)`     |

---

## 🔸 2D Loop Example
```python
for (i, j), val in np.ndenumerate(arr):
    print((i, j), val)
```

---

# 🎯 One-Liner Examples

| Function                | One-Liner Example                                                                                   | Description                                                |
|------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| `np.argmin()`          | `idx = np.argmin(a)`                                                                                | Get index of the min value in flattened array `a`          |
| `np.unravel_index()`   | `multi_idx = np.unravel_index(idx, a.shape)`  `np.unravel_index(99,(6,7,8))` here 100th index is found on a 678 matrix                                                       | Convert flat idx to (row, col)                             |
| `np.put()`             | `np.put(a, [1, 3], 9)`                                                                              | Set positions 1 and 3 in flattened `a` to 9                |
| Set min to 0           | `np.put(a, [np.argmin(a)], 0)`                                                                      | Replace min value in `a` with 0                            |
| Random set to 1        | `np.put(a, np.random.choice(a.size, 4, replace=False), 1)`                                          | Set 4 random positions in `a` to 1                         |

---

# 🎯 np.isnan(Z).all(axis=0)

## Purpose: Check which columns in array `Z` are entirely NaN.
```python
result = np.isnan(Z).all(axis=0)
print(result)  # Example: [ True False ]
```
## Special Nan Examples and more 
```python
# Comparison with nans always gives nans
0 * np.nan                 # nan
np.nan == np.nan           # False    
np.inf > np.nan            # False
np.nan - np.nan            # False
np.nan in set([np.nan])    # True
0.3 == 3 * 0.1             # False This is because floating point decimal representation issue
```

# Floor  and Ceiling 
```python
np.ceil(Z)  # 1.2 will give you 3
np.floor(Z) # 1.2 will give you 1
```

**Note**: Use `#` comments to understand code snippets. Many operations are **broadcastable** and **vectorized** for performance.


