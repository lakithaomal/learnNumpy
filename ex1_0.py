import numpy as np 

a= [1,2,3]
b= [2,1,5]

aa= np.array(a)
bb= np.array(b)

# print(a*b) 
# TypeError: can't multiply sequence by non-int of type 'list'

print(aa*bb)



