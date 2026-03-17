import pyautogui
import time

while True:
    
    x,y=pyautogui.position()
    print(f"x: {x}, y: {y}")
    time.sleep(1)
    

    """
    x: 1258, y: 416
    x: 1908, y: 864
    """