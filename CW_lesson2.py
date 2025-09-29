import cv2
import numpy as np

# image = cv2.imread('images/image.png')
# image  = cv2.resize(image, (800, 400))
# image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
# image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# image = cv2.flip(image, 1)
# image = cv2.GaussianBlur(image, (5, 5), 3)
# Рівненям блюру можуть бути тільки не парні числа.
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image, 100, 100)
# image = cv2.dilate(image, None, iterations = 3)
# kernel = cv2.ones((5, 5), np.unit8)
image = cv2.dilate(image, kernel, iterations = 1)
image = cv2.erode(image, kernel, iterations = 1)

# print(image.shape)

# # cv2.imshow('money', image)
# cv2.imshow("image", image[0:200, 0:400])

# video = cv2.VideoCapture("video/video.mp4")
video = cv2.VideoCapture(0)
# 0 для вебкамери.
while True:
    mistake, frame = video.read()
    frame = cv2.resize(frame, (800, 600))
    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cv2.waitKey(0)
cv2.destroyAllWindows()

