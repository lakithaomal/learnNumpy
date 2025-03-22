import numpy as np 

# Random Numbers 
print("--------------------")
a = np.random.rand(2)
# Gives random numbers uniformly distributed  [0.0, 1.0) 
print(a) # [0.0, 1.0) 
print(a.ndim)  # gives 1
print(a.shape) # gives (2,)
print(a.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(a.dtype) # float64


print("--------------------")
b = np.random.rand(2,4) # Not given a tuple - But just the shape 
# Gives random numbers uniformly distributed  [0.0, 1.0) 
print(b) # [0.0, 1.0) 
print(b.ndim)  # gives 2
print(b.shape) # gives (2, 4)
print(b.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(b.dtype) # float64

print("--------------------")
c = np.random.randint(7)
# Gives random int numbers uniformly distributed  [0, 7)  and is not an array
print(c) # [0.0, 1.0) 
# Does not give a dimension, shape or itemsize 

print("--------------------")
d = np.random.randint(2,4) # Not given a tuple - But just the shape 
# Gives random numbers uniformly distributed  [2, 4) 
print(d) # [0.0, 1.0) 
# Does not give a dimension, shape or itemsize 

print("--------------------")
e = np.random.randint(2,4,size=(3,5)) # Not given a tuple - But just the shape 
# Gives random numbers uniformly distributed  [2, 4) 
print(e) 
# [[2 2 3 3 3]
#  [2 3 3 2 3]
#  [3 2 2 3 3]]
print(e.ndim)  # gives 2
print(e.shape) # gives (2, 4)
print(e.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(e.dtype) # float64
 

 