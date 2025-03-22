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



















