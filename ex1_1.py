import numpy as np 

# Creating an array 
print("-----------------")
a = np.array([1,2,3,4])
print(a)
print(a.ndim)  # gives 1 
print(a.shape) # gives (4,)
print(a.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(a.dtype) # int64


print("-----------------")
b = np.array([[9.2,4.2,4.3],[100.3,4.4,2.3]])
print(b)
print(b.ndim)  # gives 2
print(b.shape) # gives (2,3)
print(b.itemsize)# gives 8
print(b.dtype) # float64

print("-----------------")
# Setting Item type
a = np.array([1,2,3,4], 'int16')
print(a)
print(a.ndim)  # gives 1 
print(a.shape) # gives (4,)
print(a.itemsize) # gives 2 - which meanst int 16 is 2 bytes # Basically gives me the number of bytes for one item 
print(a.dtype) # int64
