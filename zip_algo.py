import numpy as np
import pyautogui
import urllib.request
import requests
from bitarray import bitarray
from inputs import inputs
from inputs import keystrokes
import time
import keyboard


"""
Pseudocode:

1. Check a direction
2. Check validity - has it already been covered, and if its a numbered tile, is it one higher than the previous numbered tile
3. Check if all tiles have been filled (keep a running count), if so, then solved.
4. If no valid moves, recurse backwards. If remaining valid moves, recurse forwards.

2 angles.
1. Screenread with pyautogui + tesseract, convert to array.
2. Prescrape with beautifulsoup and solve.

Clear using keyboard.
"""

def bitarray_indexing(x,y,n):
    return x*n+y

def print_bitarray(bitarr, n):
    for i in range(n):
        print(bitarr[i*n:(i+1)*n])


def solve(map, n, bar_x = None, bar_y = None):
    
    visited = bitarray(n*n) #List of visited locations
    barrier_x = bar_x #Barriers along the x-axis (vertical barriers)
    barrier_y = bar_y #Barriers along the y-axis (horizontal barriers)
    barrier_mask = bitarray(n*n)
    ordering_list = [] #List of bitarrays. Each index corresponds to an ascending "number" tile
    currentloc = 0 #Bitarray index of starting location
    currentno = 1 #Which number we're currently on
    stack = [] #Stack containing all nodes to be visited. List of lists, matching 1 to 1 with path, each internal list containing the indexes of nodes to be visited.
    path = [] #Stack containing the path currently taken
    currentno_list = [] #Stack containing currentno

    for i in range(int(map.max())):
        x, y = np.where(map == i+1)
        temp = bitarray(n*n)
        ind = x*n + y
        ind = ind[0]
        
        temp[ind] = True

        if (i==0):
            currentloc = int(ind)
            visited[ind] = True
        
        ordering_list.append(temp)

    temp = bitarray(n*n)
    for a in ordering_list:
        temp = temp | a
    ordering_list.append(temp)

    solved = False
    ctr = 0
    backtracked = False
    remaining = n*n #Performance optimisation
    while solved == False:

        ctr += 1
        if ctr > 1000000000:
            return -1 #Failed to find a solve in acceptable number of iterations
        
       
        if backtracked == False: #If travelling forwards
            #First, find viable moves. Use bitarray masks. 1 means VIABLE
            #Viable moves by adjacency.
            viable = bitarray(n*n)
            modulo = currentloc%n
            if modulo != 0:
                viable[currentloc-1] = True
            if modulo != n-1:
                viable[currentloc+1] = True
            if not(currentloc < n):
                viable[currentloc-n] = True
            if not(currentloc >= (n*n)-n):
                viable[currentloc+n] = True
            
            #Unviable moves due to visitation
            viable = viable & ~visited

            #Unviable moves due to barriers.

            if barrier_x and barrier_y:

                floor = int(np.floor(currentloc/n))
                mod = currentloc%n

                barrier_mask[floor*n:floor*n+mod] = barrier_x[floor*(n-1):floor*(n-1)+mod]
                barrier_mask[floor*n+mod+1:(floor+1)*n] = barrier_x[floor*(n-1)+mod:((floor+1)*(n-1))]

                barrier_mask[mod:currentloc:n] = barrier_y[mod:floor*(n)+mod:n]
                barrier_mask[(floor+1)*n+mod::n] = barrier_y[(floor)*(n)+mod::n]

                viable = viable & ~barrier_mask

            
            #Unviable moves due to numbers.
            currentno_mask = ordering_list[-1] ^ ordering_list[currentno]
            viable = viable & ~currentno_mask

            #Add all viable neighbours to stack
            viable_neighbours = list(viable.search(bitarray('1')))
            if not viable_neighbours: #If no viable neighbours
                if remaining == 1: #If all points have been visited
                    solved = True
                    path.append(currentloc)
                    currentno_list.append(currentno)
                    return path, ctr #Problem solved, return the path
                else: #Dead end, need to backtrack
                    visited[currentloc] = False
                    currentloc = path.pop() #Return to previous node.
                    currentno = currentno_list.pop()
                    backtracked = True #Maybe a useful tag to show we've backtracked.
                    remaining += 1
            else:
                stack.append(viable_neighbours) #Adds a list of indexes to the stack.
        
        backtracked = False

        #Explore next node
        if not stack[-1]: #If final list in stack is empty
            visited[currentloc] = False
            currentloc = path.pop() #Return to previous node.
            currentno = currentno_list.pop()
            stack.pop()
            backtracked = True #Need to backtrack.
            remaining += 1
        else:
            path.append(currentloc) #Add current location to path
            currentno_list.append(currentno)
            currentloc = stack[-1].pop() #Travel to next location
            if ordering_list[currentno][currentloc] == 1:
                currentno += 1
            visited[currentloc] = True
            remaining -= 1            
            

        if ctr%1000000 == 0:
            print("====================")
            print_bitarray(visited, n)
            print(path)
            print(currentno_list)



    
    #Could choose to solve this problem recursively, but I'm going to use an iterative solution.




def full_solve():

    n = 6
    map = np.zeros([n,n])
    map[1][4] = 1
    map[4][4] = 2
    map[3][5] = 3
    map[0][5] = 4
    map[0][3] = 5
    map[1][1] = 6
    map[0][0] = 7
    map[2][0] = 8
    map[5][0] = 9
    map[4][1] = 10
    map[5][2] = 11
    map[5][5] = 12


    barrier_x = bitarray(n*(n-1))
    barrier_x[14] = True
    barrier_x[15] = True
    barrier_y = bitarray(n*(n-1))
    barrier_y[14] = True
    barrier_y[15] = True

    print(map)
    
    path, ctr =solve(map, n, barrier_x, barrier_y)

    ins = inputs(path,n)

    print(ins)
    print(ctr)

    #keystrokes(ins)


    return ins

""" Legacy testing code
    
if __name__ == "__main__":
    
    
    start = time.time()
    ins = full_solve()
    end = time.time()
    print(end-start)
    while not keyboard.is_pressed('8'):
        pass
    keystrokes(ins)
    print("Complete")

    
"""