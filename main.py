from zip_solve import zip_solve
import numpy as np
import pyautogui
import cv2
import keyboard
import time
from inputs import keystrokes

if __name__ == "__main__":

    config = 0 #0 for ZIP game, can expand to others later.

    if config == 0:
        print("Setup done, press 8 when puzzle appears on screen.")
        while not keyboard.is_pressed('8'):
            pass
        start_time = time.time()
        img = pyautogui.screenshot()
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        inputs, ctr = zip_solve(img)
        print(f"Solved in {time.time()-start_time}. Waiting for 2 seconds so LinkedIn doesn't throw a fit.")
        while time.time()-start_time < 2:
            pass
        keystrokes(inputs)
    else:
        print("Invalid config")
    

