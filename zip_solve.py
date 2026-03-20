import cv2 as cv
import numpy as np
import random
from matplotlib import pyplot as plt
import scipy.signal as signal
import pytesseract as pt
from cv_utils import detect_grid
import time

def zip_screenread(img, debug=False): #Takes in an image, outputs grid numbers and barriers.
    assert img is not None

    n, thresh_cropped, (x,y,width,height) = detect_grid(img, False)
    thresh2_cropped = cv.threshold(img[y:y+height, x:x+width], 100, 255, cv.THRESH_BINARY_INV)

    if debug:
        cv.imshow("", thresh2_cropped[1])
        cv.waitKey(0)

    horiz_spacing = width/(n)
    vert_spacing = height/(n)

    #Hough Circles method. Documentation on OpenCV. TLDR: Uses edge gradient information to estimate circle centres. Approximately O(n^2). Might try my own implementation at some point.
    #Utilises the fact that the digits in the "ZIP" puzzle are very nicely circled.

    circles = cv.HoughCircles(thresh_cropped[1], cv.HOUGH_GRADIENT, dp=1, minDist=(width/(n*2)), param1=200, param2 = 15, minRadius=5, maxRadius= 32)

    if debug and False:
        for (x,y,r) in circles[0]:
            cv.circle(thresh_cropped[1], (int(x),int(y)), int(r), (0, 0, 255), 2)
        
        cv.imshow("", thresh_cropped[1])
        cv.waitKey(0)
    
    digit_loc = []
    sanity = []

    if circles is not None:
        circles = circles[0].astype("int") #Integers
        
        ctr = 0
        for (x,y,r) in circles:
            r = int(r*0.9) #Slight padding
            digit = thresh_cropped[1][y-r:y+r, x-r:x+r]

            digit = cv.bitwise_not(digit)
            digit = cv.resize(digit, dsize=None, fx=4, fy=4)

            
            digit_dialated = cv.dilate(digit, np.ones((4,4), np.uint8), iterations=1)
            digit_dialated = cv.bitwise_not(digit_dialated)
            digit_dialated = cv.copyMakeBorder(digit_dialated, 30, 30, 30, 30, cv.BORDER_CONSTANT, value=255)

            if debug:
                cv.imshow("", digit_dialated)
                cv.waitKey(0)
                cv.imwrite(f"digits_training/digit{ctr}_2.png", digit)
                ctr+=1


            config = r'--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'

            digit_val = pt.image_to_string(digit_dialated, config=config)

            row_ind = (y)//vert_spacing
            col_ind = (x)//horiz_spacing

            if debug:
                print(f"Digit val: {digit_val}")
                print(f"row: {row_ind}, col: {col_ind}")

            if digit_val is not "":
                digit_loc.append((int(digit_val), int(row_ind), int(col_ind)))
                sanity.append(int(digit_val))
        
    sanity_set = set(sanity)
    if len(sanity) == max(sanity) == len(sanity_set):
        pass
    else:
        raise ValueError("Some digits were recognised incorrectly or missed.")
    
    contours, heirarchy = cv.findContours(thresh2_cropped[1], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if debug:
        
        draw = np.zeros((thresh2_cropped[1].shape[0], thresh2_cropped[1].shape[1], 3), dtype=np.uint8)

        for i in range(len(contours)):
            colour = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            cv.drawContours(draw, contours, i, colour, 2, cv.LINE_8, heirarchy, 0)
        
        cv.imshow('', draw)
        cv.waitKey(0)

        contours = [cnt for cnt in contours if len(cnt)<=50]
        for cnt in contours:
            print("Contour: ")
            print(len(cnt))
            print(cnt)

    return digit_loc, None



def zip_solve(img, debug=False):

    digit_loc, barrier_loc = zip_screenread(img, debug)
    print(digit_loc)




start = time.time()
img = cv.imread('test_screenshots/zip/test_screenshot_4.png', cv.IMREAD_GRAYSCALE)
zip_solve(img, debug=True)

print(time.time()-start)