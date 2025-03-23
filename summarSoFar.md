
# NumPy Summary

NumPy is a powerful Python library for numerical computing, built on fast, fixed-type, multidimensional arrays. It underpins many other libraries like **Pandas**, **Matplotlib**, and **scikit-learn**.

---

## 1. Basics
- **Why NumPy Over Lists?**
  - Faster due to **fixed data types** (e.g., `int8`, `int16`, `int64`)
  - **Contiguous memory**: better cache utilization and SIMD vector processing
  - Lower memory usage: avoids storing metadata like reference count, type, etc.
  - Enables efficient element-wise operations (unlike Python lists)

---

## 2. Array Creation
- `np.array([1, 2, 3], dtype='int16')`: create array with specified data type
- `np.zeros((3,3))`: create zero matrix
- `np.ones((3,3))`: create matrix of ones
- `np.identity(5)`: create identity matrix
- `np.arange(10, 50)`: array from 10 to 49
- `np.linspace(0,1,12)`: 12 evenly spaced values between 0 and 1
- `np.random.random((3,3,3))`: random 3D matrix
- `np.pad(array, pad_width=1, mode='constant', constant_values=0)`: pad array with zeros

---

## 3. Array Properties
- `.ndim`: number of dimensions
- `.shape`: shape of array
- `.itemsize`: bytes per element
- `.dtype`: data type

---

## 4. Operations
- **Arithmetic**: `+`, `-`, `*`, `/`, `**` (element-wise)
- **Comparison**: `==`, `!=`, `>`, `<`
- `np.matmul(a, b)`: matrix multiplication
- `np.sum()`, `np.add.reduce()`: sum over elements
- `np.sqrt()`, `np.emath.sqrt()`: square root (emath handles complex numbers)
- `np.floor()`, `np.ceil()`: round down/up
- `np.where(cond, x, y)`: conditional selection
- `np.intersect1d(a, b)`: common values
- `np.union1d(a, b)`: all unique values

---

## 5. Indexing & Reshaping
- `[::-1]`: reverse array
- `np.where(array != 0)`: non-zero indices
- `np.unravel_index(99, (6,7,8))`: convert flat index to coordinates
- `Z[:, [0, -1]] = 0`: modify borders
- `Z[[0, -1], :] = 0`: modify top/bottom rows
- `np.diag(1 + np.arange(4), k=-1)`: values below diagonal

---

## 6. Special Patterns
- **Checkerboard**:
  ```python
  Z = np.zeros((8,8), dtype=int)
  Z[1::2, ::2] = 1
  Z[::2, 1::2] = 1
  ```
- **Meshgrid**:
  ```python
  x, y = np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))
  ```

---

## 7. Data Type & Casting
- `x.astype(np.uint8)`: convert data type
- **Custom dtype**:
  ```python
  color = np.dtype([('r', np.ubyte), ('g', np.ubyte), ('b', np.ubyte), ('a', np.ubyte)])
  ```

---

## 8. DateTime with NumPy
- `np.datetime64('today')`, `np.timedelta64(1)`
- All July Dates:
  ```python
  Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
  ```

---

## 9. Statistical Functions
- `np.max()`, `np.min()`
- `Z.argmax()`, `Z.argmin()`
- `np.mean()`, `np.std()`
- Normalize:
  ```python
  (Z - np.mean(Z)) / np.std(Z)
  ```

---

## 10. Array Comparison
- `np.allclose(a, b)`: approximately equal
- `np.array_equal(a, b)`: exactly equal

---

## 11. Tiling & Repeating

### np.tile(array, reps)
Repeats entire array:
```python
np.tile([1,2,3], 2) → [1,2,3,1,2,3]
```

### np.repeat(array, reps)
Repeats each element:
```python
np.repeat([1,2,3], 2) → [1,1,2,2,3,3]
```

---

## 12. Bitwise & Logic
- `2 << Z >> 2`: bit shifting
- `Z**Z`: power
- `1j*Z`: complex numbers
- `Z < Z > Z`: logical chaining
- Floor Division: `//`
- Modulo: `%`

---

## 13. Sum Quirks
- Python `sum(range(5), -1)` = 9
- NumPy `sum(range(5), -1)` = 10 (sums over last axis)

---

## 14. Formatting Strings in Python

### Old Style
- `%d`, `%f`, `%.2f`, `%s`, `%x`, `%e`

### f-Strings (Python 3.6+)
```python
f"Price: {price:.2f}"
```

---

## 15. Useful Help
- `np.info('add')`: show documentation for function

---
