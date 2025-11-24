import cv2
import numpy as np
bg = np.full((400, 600, 3), (56, 98, 193), np.uint8)
cv2.rectangle(bg, (10, 10), (590, 390), (144, 245, 66), 3)
photo = cv2.imread("photo/my_photo1.jpg")
x, y, w, h = 40, 30, 130, 150
resized = cv2.resize(photo, (w, h))
bg[y:y+h, x:x+w] = resized
cv2.putText(bg, "Krasnopolski Mark", (190, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.putText(bg, "Computer Vision Student", (190, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
cv2.putText(bg, "Email: polpok172@gmail.com", (190, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (75, 75, 75), 1)
cv2.putText(bg, "Phone: +380 931061536", (190, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (75, 75, 75), 1)
cv2.putText(bg, "21/02/2010", (190, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (75, 75, 75), 1)
qr = cv2.imread("photo/qr2.jpeg")
qr = cv2.resize(qr, (100, 100))
bg[225:325, 470:570] = qr
cv2.putText(bg, "OpenCV Business Card", (150, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 0, 120), 3)
cv2.rectangle(bg, (470, 225), (570, 325), (144, 245, 66), 2)

cv2.imwrite("business_card.png", bg)
cv2.imshow("Business Card", bg)
cv2.waitKey(0)
cv2.destroyAllWindows()
