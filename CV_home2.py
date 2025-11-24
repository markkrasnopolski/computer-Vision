import cv2

image = cv2.imread('images/my_photo.jpg')
resized = cv2.resize(image, (500, 600))
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
cv2.imwrite("resized.jpg", resized)
cv2.imwrite("gray.jpg", gray)
cv2.imwrite("edges.jpg", edges)

image1 = cv2.imread('images/e-mail.jpg')
resized1 = cv2.resize(image1, (400, 200))
gray = cv2.cvtColor(resized1, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
cv2.imwrite("email_resized.jpg", resized1, [cv2.IMWRITE_JPEG_QUALITY, 50])
cv2.imwrite("email_gray.jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 50])
cv2.imwrite("email_edges.jpg", edges, [cv2.IMWRITE_JPEG_QUALITY, 50])