import numpy as np 

# MCLNS 
a = np.genfromtxt('../data/data.txt',delimiter=',')
print(a)

a= a.astype('int32')
print(a)

# Advanced Indexing

print(a>5)

print(a[a>53])

a[a>53] = 1
print(a)
# [196  75 766  75  55 999  78  76  88]
# [[   1   13   21   11 9999 9999    4    3   34    6    7    8    0    1
#      2    3    4    5]
#  [   3   42   12   33 9999 9999    4 9999    6    4    3    4    5    6
#      7    0   11   12]
#  [   1   22   33   11 9999   11    2    1 9999    0    1    2    9    8
#      7    1 9999 9999]]

# List Indexing 
print("--------------")
print(a[[1,2],[1,1]]) 
# --------------
# [42 22]


print(np.any(a>5,axis=0)) # Again checks columnwise 
# [False  True  True  True False  True False False  True  True  True  True True  True  True False  True  True]
print(np.any(a>5,axis=1)) # Again checks rowwise
# [ True  True  True]

print(np.all(a>5,axis=0)) # Again checks columnwise 
# [False  True  True  True False False False False False False False False  False False False False False False]
print(np.all(a>5,axis=1)) # Again checks rowwise
# [False False False]

print("--------------------------")
print(a)
# Boolean operators are also allowed f
print(((a>50) & (a<20)))  
# [[False False False False False False False False False False False False
#   False False False False False False]
#  [False False False False False False False False False False False False
#   False False False False False False]
#  [False False False False False False False False False False False False
#   False False False False False False]]
print(~((a>20) & (a<100)))  
# [[ True  True False  True  True  True  True  True False  True  True  True
#    True  True  True  True  True  True]
#  [ True False  True False  True  True  True  True  True  True  True  True
#    True  True  True  True  True  True]
#  [ True False False  True  True  True  True  True  True  True  True  True
#    True  True  True  True  True  True]]