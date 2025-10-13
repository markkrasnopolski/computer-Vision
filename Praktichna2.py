import cv2
import numpy as np

img = cv2.imread("images/image.png")
result = img.copy()
img = cv2.GaussianBlur(img, (5, 5), 2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])
mask_red1 = cv2.inRange(img, lower_red1, upper_red1)
mask_red2 = cv2.inRange(img, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)
mask_green = cv2.inRange(img, lower_green, upper_green)
mask_blue = cv2.inRange(img, lower_blue, upper_blue)
mask_yellow = cv2.inRange(img, lower_yellow, upper_yellow)

mask_total = cv2.bitwise_or(mask_red, mask_green)
mask_total = cv2.bitwise_or(mask_total, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_yellow)

contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def detect_shape(cnt):
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    if len(approx) == 3:
        return "Trio"
    elif len(approx) == 4:
        return "Quadro"
    elif len(approx) > 4:
        return "Oval"
    else:
        return "Else"

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
        continue
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    color = "Unknown"
    if mask_red[cy, cx] > 0:
        color = "Red"
    elif mask_green[cy, cx] > 0:
        color = "Green"
    elif mask_blue[cy, cx] > 0:
        color = "Blue"
    elif mask_yellow[cy, cx] > 0:
        color = "Yellow"

    shape = detect_shape(cnt)

    cv2.drawContours(result, [cnt], -1, (0, 0, 0), 2)
    cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
    text = f"{color}, {shape}, S={int(area)}, ({cx},{cy})"
    cv2.putText(result, text, (cx - 120, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)

cv2.imshow("Result", result)
cv2.imwrite("result.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


