import cv2
img = cv2.imread("photo/my_photo.jpg")
resized = cv2.resize(img, (500, 600))
x, y, w, h = 100, 100, 200, 200
cv2.rectangle(resized, (85,155), (380,465), (56, 98, 193), 2)
text = "Краснопольський Марк"
cv2.putText(resized, "Krasnopolski Mark", (85, 495),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
cv2.imwrite("HW_lesson_3_result.jpg", resized)
cv2.imshow("Result", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
