import cv2 as cv
import numpy as np
import random
from matplotlib import pyplot as plt
import scipy.signal as signal
import pytesseract as pt

def screenread(img, debug=False): #Takes in an image, outputs grid numbers and barriers.
    assert img is not None

    thresh2 = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)

    if debug and False:
        cv.imshow('',thresh[1])
        cv.waitKey(0)
        cv.imshow('', thresh2[1])
        cv.waitKey(0)

    contours, _ = cv.findContours(thresh2[1], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) #Finds contours from image

    if debug and False:
        
        draw = np.zeros((thresh2[1].shape[0], thresh2[1].shape[1], 3), dtype=np.uint8)

        for i in range(len(contours)):
            colour = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            cv.drawContours(draw, contours, i, colour, 2, cv.LINE_8, heirarchy, 0)
        
        cv.imshow('', draw)
        cv.waitKey(0)
        cnt = contours[0]
    
    best = None
    area_max = 0

    for c in contours: #Finds largest "Square-like" contour. Corresponds to puzzle grid.
        x, y, width, height = cv.boundingRect(c)


        area = width * height
        

        if 0.8 < width/height < 1.2 and area > area_max:
            best = (x,y,width,height)
            area_max = area
    
    x, y, width, height = best
    cropped_img = img[y:y+height, x:x+width] #Crops image to extract just the grid.

    if debug:
        print(f"Height and Width: {height}, {width}")

    if debug and False:
        cv.imshow('', cropped_img)
        cv.waitKey(0)

    thresh_cropped = cv.threshold(cropped_img, 200, 255, cv.THRESH_BINARY_INV)

    if debug:
        cv.imshow('', thresh_cropped[1])
        cv.waitKey(0)

    if debug and False:
        draw = np.zeros((cropped_img.shape[0], cropped_img.shape[1], 3), dtype=np.uint8)

        for i in range(len(new_contours)):
            colour = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            cv.drawContours(draw, new_contours, i, colour, 2, cv.LINE_8, heir, 0)
        
        cv.imshow('', draw)
        cv.waitKey(0)


    y_proj = np.mean(thresh_cropped[1], axis=0)
    x_proj = np.mean(thresh_cropped[1], axis=1)

    y_proj[y_proj<150] = 0
    x_proj[x_proj<150] = 0

    y_peaks, _ = signal.find_peaks(y_proj, distance=20)
    x_peaks, _ = signal.find_peaks(x_proj, distance=20)
    
    n = None

    if len(y_peaks) != len(x_peaks):
        print("detected y and x axis are not equal, something went wrong!")
        return -1
    else:
        n = len(y_peaks) - 1

    if debug:
        print(f"Detected grid size: {n}")

    #Hough Circles method. Documentation on OpenCV. TLDR: Uses edge gradient information to estimate circle centres. Approximately O(n^2). Might try my own implementation at some point.
    #Utilises the fact that the digits in the "ZIP" puzzle are very nicely circled.

    circles = cv.HoughCircles(thresh_cropped[1], cv.HOUGH_GRADIENT, dp=1, minDist=(width/(n*2)), param1=200, param2 = 15, minRadius=5, maxRadius= 32)

    if debug and False:
        for (x,y,r) in circles[0]:
            cv.circle(thresh_cropped[1], (int(x),int(y)), int(r), (0, 0, 255), 2)
        
        cv.imshow("", thresh_cropped[1])
        cv.waitKey(0)

    if circles is not None:
        circles = circles[0].astype("int") #Integers
        ctr = 0
        for (x,y,r) in circles:
            r = int(r*0.7) #Slight padding
            digit = thresh_cropped[1][y-r:y+r, x-r:x+r]

            digit = cv.bitwise_not(digit)
            digit = cv.resize(digit, dsize=None, fx=4, fy=4)

            digit_dialated = cv.dilate(digit, np.ones((4,4), np.uint8), iterations=1)

            cv.imshow("", digit_dialated)
            cv.waitKey(0)

            cv.imwrite(f"digit{ctr}.png", digit_dialated)

            config = r'--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'

            digit_val = None

            digit_val = pt.image_to_string(digit_dialated, config=config)

            print(digit_val)


            cv.imshow("", digit)
            cv.waitKey(0)
            ctr+=1






img = cv.imread('test_screenshot_2.png', cv.IMREAD_GRAYSCALE)
screenread(img, debug=True)