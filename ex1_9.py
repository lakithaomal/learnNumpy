import numpy as np 

# Linear Algebra 
a = np.ones((2,3))
print(a)

b = np.full((3,2), 2)
print(b)

# print(a*b) # Element wise muliplication does not work - 

# Matrix multiplication works though 
print(np.matmul(a,b))
c = np.matmul(a,b)


d  = np.array([[1,2,3],[2,3,4],[3,4,5]])

print("---------")
# Get the determinant 
print(np.linalg.det(d))

# It can igon values 
print(np.linalg.eig(d))

# It can do inverse 
print(np.linalg.inv(d))






