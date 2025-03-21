import numpy as np 

# Creating an array 
print("-----------------")
a = np.array([\
                [\
                    [1,2,3,4,5,6,7],\
                        [8,9,10,11,12,13,14],\
                            [1,2,3,4,5,6,7],\
                                [8,9,10,11,12,13,14]\
                ],
                [\
                    [1,2,3,4,5,6,7],\
                        [8,9,10,11,12,13,14],\
                            [1,2,3,4,5,6,7],\
                                [8,9,10,11,12,13,14]\
                ],

            ])
print(a)
print(a.ndim)  # gives 3
print(a.shape) # gives (2, 4, 7)
print(a.itemsize) # gives 8 # Basically gives me the number of bytes for one item 
print(a.dtype) # int64


print(a[0,1,4]) # Gives 12 
a[0,1,4] = 199
print(a)
# [[[  1   2   3   4   5   6   7]
#   [  8   9  10  11 199  13  14]
#   [  1   2   3   4   5   6   7]
#   [  8   9  10  11  12  13  14]]

#  [[  1   2   3   4   5   6   7]
#   [  8   9  10  11  12  13  14]
#   [  1   2   3   4   5   6   7]
#   [  8   9  10  11  12  13  14]]]
print(a[0,1,4]) # Gives 199




