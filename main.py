import cv2
import numpy as np
import imutils
import requests

# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
url = "http://192.168.0.6:8080/shot.jpg"

while True:
    # FROM MOBILE
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)
    #
    imageFrame = img
    # imageFrame = cv2.imread("1.jpg")


    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    # grayFrame = cv2.cvtColor(imageFrame, cv2.COLOR_RGB2GRAY)

    # obtaining white mask
    # sensitivity = 15
    white_lower = np.array([0, 0, 0], dtype=np.uint8)
    white_upper = np.array([127, 96, 96], dtype=np.uint8)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

    # test
    cv2.imshow("white mask", white_mask)

    kernal = np.ones((5, 5), "uint8")

    # For white color
    white_mask = cv2.dilate(white_mask, kernal)
    res_white = cv2.bitwise_and(imageFrame, imageFrame, mask=white_mask)

    # Creating contour to track white color
    contours, hierarchy = cv2.findContours(white_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    index = 0
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if index == 0:
            largest = area
        if area > largest:
            largest = area
        else:
            index += 1
            continue

        x, y, w, h = cv2.boundingRect(contour)
        imageFrame = cv2.rectangle(imageFrame, (x, y),
                                   (x + w, y + h),
                                   (0, 0, 255), 2)

        cv2.putText(imageFrame, "Arena", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255))
        index += 1

        # if area > 300:
        #     x, y, w, h = cv2.boundingRect(contour)
        #     imageFrame = cv2.rectangle(imageFrame, (x, y),
        #                                (x + w, y + h),
        #                                (0, 0, 255), 2)
        #
        #     cv2.putText(imageFrame, "White Colour", (x, y),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        #                 (0, 0, 255))

    # print grayscale
    # grayFrame = cv2.cvtColor(imageFrame, cv2.COLOR_RGB2GRAY)

    # Program Termination
    # cv2.imshow("Gray", grayFrame)
    cv2.imshow("white mask dilated", white_mask)
    cv2.imshow("bit wise anded", white_mask)
    cv2.imshow("HSV", hsvFrame)
    cv2.imshow("BGR", imageFrame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
