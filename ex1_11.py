import numpy as np 

# Reorganizing arrays
 
a = np.array([[1,2,3],[4,5,6]])
print(np.shape(a))

b = a.reshape(1,6)
print(b)
# [[1 2 3 4 5 6]]

c = a.reshape(3,2,1)
print(c)
# [[[1]
#   [2]]

#  [[3]
#   [4]]

#  [[5]
#   [6]]]

# You can also stack 

v1 = np.array([1,2,3,4])
print(v1)
v2 = np.array([4,5,6,7])
print(v2)

v3 = np.vstack([v1,v2])
print(v3)
# [[1 2 3 4]
#  [4 5 6 7]]

v4 = np.hstack([v1,v2])
print(v4) # [1 2 3 4 4 5 6 7]
# Reshape 

h1 = np.ones((2,4))
print(h1)

h2 = np.zeros((2,2))
print(h2)

h3 = np.hstack([h1,h2,h1])
print(h3)