import numpy as np 

# Other Functions 
print("--------------------")
# Exersize 
a = np.zeros((5,5) ,'int16')
a[:,0] = 1 
a[:,-1] = 1 
a[0,:] = 1 
a[-1,:] = 1 
a[2,2] = 9 
print(a)


# Way 2 
b = np.ones((5,5) ,'int16')
c = np.zeros((3,3) ,'int16')
c[1,1] = 9 
print(b)
print(c)
b[1:4,1:4] = c
print(b)


## Copying Arrays 
print("-------------------")
d = np.array([1,1,2,2,3])
e=d  # Be carefule - This is essentially pointing to a memory location
e[0] = 10000

print(d) 
print(e)


f = d.copy()
f[2] = 988

print(d)
print(e)
print(f)

