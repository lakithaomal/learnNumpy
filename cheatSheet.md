## Importing NumPy
- `import numpy as np` – Imports the NumPy library and aliases it as `np` for convenience.

---

### Creating Arrays
- `np.array([1, 2, 3])` – Creates a 1D array from a Python list.
- `np.array([[1, 2], [3, 4]])` – Creates a 2D array (matrix) from nested lists.
- `np.zeros((2, 3))` – Creates a 2x3 array filled with zeros.
- `np.ones((2, 3))` – Creates a 2x3 array filled with ones.
- `np.empty((2, 3))` – Creates a 2x3 array with uninitialized (random) values.
- `np.arange(0, 10, 2)` – Creates a 1D array with values from 0 to 10 (exclusive) in steps of 2.
- `np.linspace(0, 1, 5)` – Creates 5 evenly spaced numbers between 0 and 1.

---

### Array Properties
- `array.shape` – Returns the dimensions (rows, columns) of the array.
- `array.ndim` – Returns the number of dimensions of the array.
- `array.size` – Returns the total number of elements in the array.
- `array.dtype` – Returns the data type of the array elements.

---

### Reshaping and Transposing
- `array.reshape((rows, cols))` – Changes the shape of the array without changing its data.
- `array.T` – Transposes the array (rows become columns and vice versa).

---

### Indexing and Slicing
- `array[i, j]` – Accesses the element at row `i` and column `j`.
- `array[i, :]` – Selects all columns of row `i`.
- `array[:, j]` – Selects all rows of column `j`.
- `array[i1:i2, j1:j2]` – Selects a subarray from rows `i1` to `i2` and columns `j1` to `j2`.

---

### Array Operations
- `array1 + array2` – Adds corresponding elements of two arrays.
- `array1 - array2` – Subtracts elements of one array from another.
- `array1 * array2` – Multiplies corresponding elements of two arrays.
- `array1 / array2` – Divides elements of one array by another element-wise.
- `np.dot(array1, array2)` – Computes the dot (matrix) product of two arrays.

---

### Statistical Methods
- `array.min()` – Returns the smallest element in the array.
- `array.max()` – Returns the largest element in the array.
- `array.sum()` – Returns the sum of all elements in the array.
- `array.mean()` – Returns the average (mean) value of the elements.
- `array.std()` – Returns the standard deviation of the elements.

---

### Random Module
- `np.random.rand(size)` – Generates random floats in [0, 1) with specified size.
- `np.random.randint(low, high, size)` – Generates random integers between `low` and `high`.
- `np.random.randn(size)` – Generates random samples from a standard normal distribution.
