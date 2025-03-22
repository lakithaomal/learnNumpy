import numpy as np 

# Creating an array 
print("-----------------")
a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a)
print(a.ndim)  # gives 1 
print(a.shape) # gives (4,)
print(a.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(a.dtype) # int64

print(a[1,5])  # Gives 13 

print(a[1,-1]) # Gives 14 

print(a[:,2]) # Gives the 2nd column from all rows
# [ 3 10]

print(a[1,:]) # Gives the 1st row from all columns
# [ 8  9 10 11 12 13 14]

print(a[0,1:6:2]) # gives the zeroth column going from 1 to 6th index incrementing by 2.
#[2 4 6]          # This is inclusinve

print(a[0,1:-1:2]) # gives the zeroth column going from 1 to 1st to last index incrementing by 2.
#[2 4 6]          # This is inclusinve

# Assignment 
a[1,5] = 500
print(a)
# [[  1   2   3   4   5   6   7]
#  [  8   9  10  11  12 500  14]]

a[:,2] = [324,342]
print(a)
# [[  1   2 324   4   5   6   7]
#  [  8   9 342  11  12 500  14]]
print("-----------")
print(a[0,0:5]) # the index of 5 is not included - Something to remember
# Its start index , end index and step size 
# [  1   2 324   4   5]

print(a[0,0:-1]) # the index of -1 is not included - Something to remember
# Its start index , end index and step size 
# [  1   2 324   4   5   6]

# Changing Columns 
a[:,2] = [1,2]
print(a) 
# [[  1   2   1   4   5   6   7]
#  [  8   9   2  11  12 500  14]]

# 3D Example 
