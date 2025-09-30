# import cv2
# import numpy as np
#
# img = np.zeros((500,400,3), np.uint8)
# img[:] = 56, 98, 193
# rgb = bgr

# img[100:150, 200:250] = 56, 98, 193
#
# cv2.rectangle(img, (100,100),(200,200),(56, 98, 193),2)
#
# cv2.line(img,(100,100),(200,200),(56, 98, 193),2)

# print(img.shape)
# cv2.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (56, 98, 193), 2)
# cv2.line(img, (0, img.shape[1] // 2, 0), (img.shape[1] // 2, img.shape[0]), (56, 98, 193), 2)

# cv2.circle(img, (200, 200), 50, (56, 98, 193), -1)
# cv2.putText(img, "Krasnopolski Mark", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 98, 193))
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()