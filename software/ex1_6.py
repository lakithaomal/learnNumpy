import numpy as np 

# Other Functions 
print("--------------------")
a = np.identity(5) # Only need one input - Identity matrices are square
print(a) # 
# [[1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]]
print(a.ndim)  # gives 2
print(a.shape) # gives (5, 5)
print(a.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(a.dtype) # float64

print("--------------------")
b = np.array([[1,2,3]]) # 
print(b) # 
# [[1 2 3]]
print(b.ndim)  # gives 2
print(b.shape) # gives (1,3)
print(b.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(b.dtype) # float64
# This is a 3D vector - 


print("--------------------")
c = np.repeat(b,4) # In
# What happens internally:
# Flatten b to: [1, 2, 3]
# Repeat each element 4 times:
# 1 → 1 1 1 1
# 2 → 2 2 2 2
# 3 → 3 3 3 3

print(c) # 
# [1 1 1 1 2 2 2 2 3 3 3 3]
print(c.ndim)  # gives 1
print(c.shape) # gives (12)
print(c.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(c.dtype) # float64
# This is a 3D vector - 


# Repeat Rows 
print("--------------------")
d = np.repeat(b,4,0) # 
# In this case the array is reapeating 4 times interms of row index (0)

print(d) # 
# [[1 2 3]
#  [1 2 3]
#  [1 2 3]
#  [1 2 3]]
print(d.ndim)  # gives 2
print(d.shape) # gives (4, 3)
print(d.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(d.dtype) # float64
# This is a 3D vector - 

# Repeat Columns 
print("--------------------")
e = np.repeat(b,6,1) # 
# In this case the array is reapeating 4 times interms of column index (1)

print(e) # 
# [[1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3]]
print(e.ndim)  # gives 2
print(e.shape) # gives (1, 18)
print(e.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(e.dtype) # float64
# This is a 3D vector - 