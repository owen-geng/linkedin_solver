import pyautogui
import numpy as np

def inputs(path,n):
    
    path.append(-1) #Marks the end
    input_list = []

    for ind, num in enumerate(path[:-1]):
        diff = num-path[ind+1]
        if diff == -n:
            input_list.append('down')
        elif diff == n:
            input_list.append('up')
        elif diff == 1:
            input_list.append('left')
        elif diff == -1:
            input_list.append('right')
        else:
            return input_list
    
    return input_list

def keystrokes(inputs):
    pyautogui.press(inputs)