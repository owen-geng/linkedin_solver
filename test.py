from bitarray import bitarray
import numpy as np

def print_bitarray(bitarr, n):
    for i in range(n):
        print(bitarr[i*n:(i+1)*n])
"""
n=4
test = bitarray(n*n)
for ind in range(0, n*n, n):
    
    test[ind+int((ind/4))] = True

print_bitarray(test, n)


loc = 6
floor = int(np.floor(loc/n))

test2 = test[floor*n:(floor*n)+n]
print(len(test2))
print(test2)

test3 = test[loc%n::n]
print(len(test3))
print(test3)
print("========")
test4 = bitarray(n*n)

test[2] = True
test[6] = True
test[14] = True
print_bitarray(test, n)
print("========")
test4[loc%n::n] = test[loc%n::n]
print_bitarray(test4,n)
"""
n=7
barrier_mask = bitarray(n*n)

currentloc = 0
floor = int(np.floor(currentloc/n))
mod = currentloc%n

barrier_x = bitarray(n*(n-1))
barrier_x[12] = True
barrier_x[14] = True
barrier_x[15] = True
barrier_x[17] = True

barrier_y = bitarray(n*(n-1))
barrier_y[8] = True
barrier_y[9] = True
barrier_y[11] = True
barrier_y[12] = True


for (i) in range(0,n*(n-1), n-1):
    print(barrier_x[i:(i+n-1)])

print("==========")
for i in range(0, n*(n-1), n):
    print(barrier_y[i:(i+n)])

print("========")
barrier_mask[floor*n:floor*n+mod] = barrier_x[floor*(n-1):floor*(n-1)+mod]
barrier_mask[floor*n+mod+1:(floor+1)*n] = barrier_x[floor*(n-1)+mod:((floor+1)*(n-1))]

barrier_mask[mod:currentloc:n] = barrier_y[mod:floor*(n)+mod:n]
barrier_mask[(floor+1)*n+mod::n] = barrier_y[(floor)*(n)+mod::n]
print("======")
print(barrier_mask[(floor+1)*n+mod::n])
print(barrier_y[(floor)*(n)+mod::n])
print("=======")
print_bitarray(barrier_mask, n)