from bitarray import bitarray

def cca(barr, n):
    init = barr.find(bitarray('0'))
    ctr = 0
    component_found = False
    while ctr < n*n:
        ctr += 1
        
