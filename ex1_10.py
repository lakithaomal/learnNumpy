import numpy as np 

# Statistics 
a = np.array([[1,2,3],[4,5,6]])

print(np.min(a)) # gives 1 
print(np.max(a)) # gives 6 

print(np.min(a,axis=0)) # gives [1 2 3]
print(np.min(a,axis=1)) # gives [1 4]

print(np.max(a,axis=0)) # gives [4 5 6]
print(np.max(a,axis=1)) # gives [3 6]

print(np.sum(a)) # gives 1 
print(np.sum(a,0)) # gives for each row [5 7 ] - This is for column wise 
print(np.sum(a,1)) # gives for each row [6, 15 ]- This is for row - its flipped 




