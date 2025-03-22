# learnNumpy

Numpy is the base for multiple other libraries - For example Pandas. 


## EX 1: 
EX1 is based off of this [video](https://www.youtube.com/watch?v=QUT1VHiLmmI&t=864s)


## What is numpy 

It is a multidimensianal array library. 


1d, 2d or upto nd. 

Why not lists. Numpy arrays are much faster than lists. The reson being numpy uses fixed types. 

In a list the int, 5 is represented as a long which is 8 bytes. However in numpy we can go down to int8 which is one byte (default is int64 (4 bytes)). Further a list a lot more information is required.
- Size             - Long (4 bytes)
- Reference Count  - Long (8 bytes)
- Object Type      - Long (8 bytes)
- Object Value     - Long (8 bytes)

Becase numpy is less bytes and the computer reads it much faster. Also numpy douesnt require type checking - and also numpy utilizeds continous memory. Unlike lists  which saves items which are scattered around. 

Benifits of continous memory
- SIMD Vector processing (Single instuction Mulitple data) - Performs computation all contous memory locations 
- Effective Cache Utilization  Cache is fast memory.

Both lists and numpy you can do Insertion, Deletion, Appending and Concatination. But numpy arrays can do a lot more. 

## Lists Vs Numpy - Execution
```
import numpy as np 

a= [1,2,3]
b= [2,1,5]

aa= np.array(a)
bb= np.array(b)

# print(a*b) 
# TypeError: can't multiply sequence by non-int of type 'list'

print(aa*bb)
# Gives [ 2  2 15]
```

## Applications of Numpy 


- Mathematics
- Plottiting (Matplotlib)
- Backend (Pandas: Handles data frames for faster data manipulations, Connect 4: Game boards, Digital Photography: Image processing): NumPy powers the backend calculations or data handling in various tools
- Machine Learning: Similar to tensors 


## Installation 
`pip3 install numpy`

## Various Basic Commands 

```
a = np.array([1,2,3,4], 'int16') # Setting up for int 16 
print(a.ndim)  # gives Dimensions 1 
print(a.shape) # gives (4,)
print(a.itemsize) # gives 2 - which meanst int 16 is 2 bytes # Basically gives me the number of bytes for one item 
print(a.dtype) # int64
```


## 1. Old-Style Formatting (% formatting)

| Placeholder | Type of Value      | Example                               | Output                  |
|-------------|--------------------|---------------------------------------|-------------------------|
| `%d`        | Integer            | `"I have %d apples" % 5`              | `"I have 5 apples"`     |
| `%f`        | Float (decimal)    | `"Price: %f" % 3.14`                  | `"Price: 3.140000"`     |
| `%.2f`      | Float (2 decimals) | `"Price: %.2f" % 3.1415`              | `"Price: 3.14"`         |
| `%s`        | String             | `"Hello, %s" % "Alice"`               | `"Hello, Alice"`        |
| `%x`        | Hexadecimal (int)  | `"Hex: %x" % 255`                     | `"Hex: ff"`             |
| `%o`        | Octal (int)        | `"Octal: %o" % 8`                     | `"Octal: 10"`           |
| `%e`        | Scientific float   | `"Sci: %e" % 12345.678`               | `"Sci: 1.234568e+04"`   |

---

## 2. Modern Formatting (f-Strings, Python 3.6+)

```python
apples = 5
price = 3.1415
name = "Alice"

print(f"I have {apples} apples")             # integer
print(f"Price: {price:.2f}")                 # float with 2 decimal places
print(f"Hello, {name}")                      # string
print(f"Hex: {255:x}")                       # hex
print(f"Sci: {12345.678:e}")                 # scientific notation

```

### Hints 
- asking for help `print(np.info('add'))`
- going from x to y `Z = np.arange(10,50)`
- reversing the grid `q08 = q07[::-1]` 
- non zero indices q10 `np.where(q10!=0)` or `q10.nonzero()`
- Identity eye or identity `np.identity(5)`
- Random Matrix `np.random.random((3,3,3))`
- Max and Min `print(x.max())` and `print(x.min())`
- Array Borders `np.pad(Z, pad_width=1, mode='constant', constant_values=0)`
  ```
  Z = np.ones((5,5))
  Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
  print(Z)

  # Using fancy indexing
  Z[:, [0, -1]] = 0
  Z[[0, -1], :] = 0
  print(Z)
  ```
  - Comparison with nans always gives nans
    - 0 * np.nan
    - np.nan == np.nan        = nan
    - np.inf > np.nan         = False
    - np.nan - np.nan         = False 
    - np.nan in set([np.nan]) = True 
    - 0.3 == 3 * 0.1          = False This is because floating point decimal representation issue
- Just below the diagonal `np.diag(1+np.arange(4),k=-1)`

-  zeros Matrix `np.zeros(64) ` creates an array with 64 elements. Not 64 * 64
- Checkerboard
    ```
    Z = np.zeros((8,8),dtype=int)
    Z[1::2,::2] = 1
    Z[::2,1::2] = 1
    ```
- Get Specific index   `np.unravel_index(99,(6,7,8))` here 100th index is found on a 6*7*8 matrix 

# NumPy `tile()` Function

The `np.tile()` function in NumPy is used to **repeat an array** along specified axes, like tiling a pattern across a larger space.

---

## Syntax
```python
numpy.tile(array, reps)
```
- **`array`**: The input array to repeat.
- **`reps`**: An integer or tuple indicating how many times to repeat along each axis.

---

## Example 1: Repeat a 1D Array
```python
import numpy as np

a = np.array([1, 2, 3])
result = np.tile(a, 2)

print(result)  # Output: [1 2 3 1 2 3]
```
**Explanation**: The array `[1 2 3]` is repeated **2 times**.

---

## Example 2: Repeat a 2D Array Along Both Axes
```python
a = np.array([[1, 2], [3, 4]])
result = np.tile(a, (2, 3))

print(result)
```

**Output**:
```
[[1 2 1 2 1 2]
 [3 4 3 4 3 4]
 [1 2 1 2 1 2]
 [3 4 3 4 3 4]]
```

**Explanation**:
- The array is repeated **2 times vertically** (rows),
- And **3 times horizontally** (columns).

---

## Visual Concept
Think of a small tile (array) and covering a big floor (larger array) by **repeating it** in rows and columns.

---

## Use Cases
- Creating larger patterns from small arrays.
- Preparing data for broadcasting.
- Synthetic dataset creation or data augmentation.

---

## Bonus: Difference Between `tile()` and `repeat()`
| Function       | Purpose                                  |
|----------------|------------------------------------------|
| `np.tile()`    | Repeats the **whole array**              |
| `np.repeat()`  | Repeats **each element** individually    |

Example:
```python
a = np.array([1, 2, 3])
print(np.tile(a, 2))    # [1 2 3 1 2 3]
print(np.repeat(a, 2))  # [1 1 2 2 3 3]
```

- Normalizing a matrix `(Z - np.mean (Z)) / (np.std (Z))`
- Casting numpy arrays `x.astype(np.uint8)`
- Creating a new data type
  ``` 
    color = np.dtype([("r", np.ubyte),
                      ("g", np.ubyte),
                      ("b", np.ubyte),
                      ("a", np.ubyte)])
  ```
- Matrix Multiplication `np.matmul(x,y)`




