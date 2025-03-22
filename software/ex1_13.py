import numpy as np 

# MCLNS 
a = np.full((6,5),1)
print(a)
np.arange(0, 31) 
# How to do a list of 1: n 

c = 0 
for x in range(6):
    for y in range(5): 
        a[x,y] = c+ 1
        print(a)
        c = c+1 
        print(c)



print(a[2:4,0:2]) # Remember inclusion 

print(a[[0,1,2,3],[1,2,3,4]])   # Remember inclusion 

print(a[[0,4,5],:][:,3:]) # 3 onwards 


a  = np.arange(1, 31) 
a = a.reshape(6,5)
print(a)