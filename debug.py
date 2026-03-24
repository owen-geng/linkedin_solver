from zip_solve import zip_solve
import numpy as np
import pyautogui
import cv2
import keyboard
import time
from inputs import keystrokes

if __name__ == "__main__":

    img = cv2.imread("test_screenshots/zip/test_screenshot_5.png")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(img)
    zip_solve(img, debug=True)
    

