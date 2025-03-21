import numpy as np 

# Math Functions
a = np.array([1,2,3,4,5,6])
# Gives random numbers uniformly distributed  [0.0, 1.0) 
print(a) #
print(a.ndim)  # gives 1
print(a.shape) # gives (2,)
print(a.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(a.dtype) # float64

print("--------------------")
print("a+2 = " + str(a+2))
print("a-2 = " + str(a-2))
print("a/2 = " + str(a/2))
a += 2 # a = a + 2
print("a again = " + str(a))
a += 2 # a = a + 2
print("a again = " + str(a))

b = np.array([1,1,0,3,1,2])

print("a+b = " + str(a+b))

print("a**2 = " + str(a**2)) # This is power

print("cos(a) = " + str(np.cos(a))) # Gives cosine

print(np.cos(90)) # This uses radians 