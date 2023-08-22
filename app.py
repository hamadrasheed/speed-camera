import cv2
import numpy as np
from time import sleep
from tracker import *

# cap = cv2.VideoCapture('h1.MOV')
# cap = cv2.VideoCapture('h6.mp4')
cap = cv2.VideoCapture('video.mp4')
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
tracker = EuclideanDistTracker()

while True:

    ret , frame = cap.read()
    height, width, _ = frame.shape

    print(height, width)

    # Naming a window
    # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    # # Using resizeWindow()
    # cv2.resizeWindow("Frame", 1000, 800)
    
    # roi = frame[1100:1400, 300:850]
    roi = frame[450:650, 100:1050]
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(roi, [cnt], -1, (0,255,0), 2)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 3)


    # cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    key  = cv2.waitKey(30)

    if key == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
