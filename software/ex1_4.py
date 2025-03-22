import numpy as np 

# Initializing arrays 
a = np.zeros(5)
print(a)
print(a.ndim)  # gives 1
print(a.shape) # gives (5,)
print(a.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(a.dtype) # float64

print("-------------------------")
b = np.zeros((2,3)) # The input should be a tuple
print(b)
print(b.ndim)  # gives 2
print(b.shape) # gives (2,3)
print(b.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(b.dtype) # float64

print("-------------------------")
c = np.ones(4)
print(c)
print(c.ndim)  # gives 1
print(c.shape) # gives (4,)
print(c.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(c.dtype) # float64


print("-------------------------")
d = np.ones((4,2), dtype='int32')
# d = np.ones((4,2), 'int32') # Same as the earlier line 
print(d)
print(d.ndim)  # gives 2
print(d.shape) # gives (4,2)
print(d.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(d.dtype) # float64

print("-------------------------")
# A fill functions 
e = np.full(2, 88)
print(e) #[88 88]
print(e.ndim)  # gives 1
print(e.shape) # gives (2,)
print(e.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(e.dtype) # int64


print("-------------------------")
# A fill functions 
f = np.full((2,5), 77)
print(f) 
# [[77 77 77 77 77]
#  [77 77 77 77 77]]
print(f.ndim)  # gives 2
print(f.shape) # gives (2,5)
print(f.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(f.dtype) # int64


print("-------------------------")
# A fill like functions - Makes an array of the shape of 
f = np.full_like(f, 66)
print(f) 
# [[77 77 77 77 77]
#  [77 77 77 77 77]]
print(f.ndim)  # gives 2
print(f.shape) # gives (2,5)
print(f.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(f.dtype) # int64