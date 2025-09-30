import cv2

image = cv2.imread('images/my_photo.jpg')
image  = cv2.resize(image, (800, 400))

cv2.rectangle(image, (790,35),(590,235),(56, 98, 193),2)
cv2.putText(image, "Krasnopolski Mark", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 98, 193))


cv2.waitKey(0)
cv2.destroyAllWindows()