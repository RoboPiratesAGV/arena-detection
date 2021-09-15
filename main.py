import cv2
import numpy as np
import imutils
import requests

while True:
    imageFrame = cv2.imread("1.jpg")
    imageFrame = imutils.resize(imageFrame, width=1000, height=1800)

    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # obtaining white mask
    # sensitivity = 15
    white_lower = np.array([0, 0, 0], dtype=np.uint8)
    white_upper = np.array([0, 0, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

    kernal = np.ones((5, 5), "uint8")

    # For white color
    white_mask = cv2.dilate(white_mask, kernal)
    res_white = cv2.bitwise_and(imageFrame, imageFrame, mask=white_mask)

    # Creating contour to track white color
    contours, hierarchy = cv2.findContours(white_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            cv2.putText(imageFrame, "White Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))

    # Program Termination
    cv2.imshow("mask", white_mask)
    cv2.imshow("HSV", hsvFrame)
    cv2.imshow("BGR", imageFrame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
