import cv2
import numpy as np

img = cv2.imread("images/shapes.png")
img_copy = img.copy()

blur = cv2.GaussianBlur(img, (5, 5), 2)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

lower = np.array([0, 50, 50])
upper = np.array([179, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

res = cv2.bitwise_and(img, img, mask=mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            aspect_ratio = round(w / h, 2)
            compactness = round(4 * np.pi * area / (perimeter ** 2), 2)

            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            sides = len(approx)

            if sides == 3:
                shape = "triangle"
            elif sides == 4:
                shape = "quadro"
            else:
                if compactness > 0.85:
                    shape = "circle"
                else:
                    hull = cv2.convexHull(cnt, returnPoints=False)
                    if hull is not None and len(hull) > 3:
                        defects = cv2.convexityDefects(cnt, hull)
                        if defects is not None:
                            num_defects = len(defects)
                            if num_defects >= 5 and compactness < 0.65:
                                shape = "star"
                            else:
                                shape = "other"
                        else:
                            shape = "other"
                    else:
                        shape = "other"

            cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(img_copy, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(img_copy, f"S:{int(area)} P:{int(perimeter)}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img_copy, f"AR:{aspect_ratio} C:{compactness}",
                        (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img_copy, f"Shape:{shape}",
                        (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

cv2.imshow("Mask", mask)
cv2.imshow("Result", img_copy)
cv2.imwrite("result.jpg", img_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()
