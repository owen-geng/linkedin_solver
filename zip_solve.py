import cv2 as cv
import numpy as np
from bitarray import bitarray
import random
from cv_utils import detect_grid
from zip_inference import predict_digit
from zip_algo import print_bitarray
from zip_algo import solve
from zip_algo import inputs

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
            r = int(r*0.7) #Slight padding
            digit = thresh_cropped[1][y-r:y+r, x-r:x+r]
            digit = cv.bitwise_not(digit)
            digit = cv.resize(digit, dsize=None, fx=4, fy=4)            
            digit_dialated = cv.dilate(digit, np.ones((4,4), np.uint8), iterations=1)
            
            if debug and False:
                cv.imshow("", digit_dialated)
                cv.waitKey(0)
                ctr+=1


            digit_val, confidence = predict_digit(digit_dialated)

            row_ind = (y)//vert_spacing
            col_ind = (x)//horiz_spacing

            if debug and False:
                print(f"Digit val: {digit_val}")
                print(f"Confidence: {confidence}")
                print(f"row: {row_ind}, col: {col_ind}")

            if digit_val != "":
                digit_loc.append((int(digit_val), int(row_ind), int(col_ind)))
                sanity.append(int(digit_val))
        
    sanity_set = set(sanity)
    if len(sanity) == max(sanity) == len(sanity_set):
        pass
    else:
        raise ValueError("Some digits were recognised incorrectly or missed.")
    

    #Digit locating and recognition complete. Now for barrier recognition.
    
    contours, heirarchy = cv.findContours(thresh2_cropped[1], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if debug and False:
        
        draw = np.zeros((thresh2_cropped[1].shape[0], thresh2_cropped[1].shape[1], 3), dtype=np.uint8)

        for i in range(len(contours)):
            colour = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            cv.drawContours(draw, contours, i, colour, 2, cv.LINE_8, heirarchy, 0)
        
        cv.imshow('', draw)
        cv.waitKey(0)

        
    h_img, w_img = thresh2_cropped[1].shape
    COVERAGE_THRESHOLD = 0.3
    strip_half = max(2, int(min(vert_spacing, horiz_spacing) * 0.15))

    barrier_x = bitarray(n * n)  # vertical barriers:   barrier_x[row*n+col] between (row,col)-(row,col+1), last col always 0
    barrier_y = bitarray(n * n)  # horizontal barriers: barrier_y[row*n+col] between (row,col)-(row+1,col), last row always 0
    barrier_x.setall(0)
    barrier_y.setall(0)

    contour_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv.drawContours(contour_mask, contours, -1, 255, thickness=cv.FILLED)

    for r in range(n - 1):
        edge_y = int(round((r + 1) * vert_spacing))
        y0 = max(0, edge_y - strip_half)
        y1 = min(h_img, edge_y + strip_half)
        for col in range(n):
            x0 = int(round(col * horiz_spacing))
            x1 = int(round((col + 1) * horiz_spacing))
            strip = contour_mask[y0:y1, x0:x1]
            if strip.sum() / (strip.size * 255 + 1e-9) >= COVERAGE_THRESHOLD:
                barrier_y[r * n + col] = True

    for c in range(n - 1):
        edge_x = int(round((c + 1) * horiz_spacing))
        x0 = max(0, edge_x - strip_half)
        x1 = min(w_img, edge_x + strip_half)
        for row in range(n):
            y0 = int(round(row * vert_spacing))
            y1 = int(round((row + 1) * vert_spacing))
            strip = contour_mask[y0:y1, x0:x1]
            if strip.sum() / (strip.size * 255 + 1e-9) >= COVERAGE_THRESHOLD:
                barrier_x[row * n + c] = True

    if debug:
        print("barrier_x:")
        print_bitarray(barrier_x, n)
        print("barrier_y:")
        print_bitarray(barrier_y, n)

    return digit_loc, (barrier_x, barrier_y), n



def zip_solve(img, debug=False):

    digit_loc, (barrier_x, barrier_y), n = zip_screenread(img, debug)
    map = map = np.zeros([n,n])
    for digit in digit_loc:
        map[digit[1]][digit[2]] = digit[0]
    # solve() expects barrier_x as n*(n-1) (strip last col) and barrier_y as n*(n-1) (strip last row)
    bx = bitarray(n * (n - 1))
    for r in range(n):
        bx[r*(n-1):(r+1)*(n-1)] = barrier_x[r*n:r*n+(n-1)]
    by = barrier_y[:n * (n - 1)]
    path, ctr = solve(map, n, bx, by)

    return inputs(path, n), ctr

