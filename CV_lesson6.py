import cv2

vid = cv2.VideoCapture(0)

ret, frame1 = vid.read()

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (5, 5), 5)
gray1 = cv2.convertScaleAbs(gray1, alpha=1.5, beta=10)

while True:
    ret, frame2= vid.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 5)
    gray2 = cv2.convertScaleAbs(gray2, alpha=1.5, beta=10)

    if not ret:
        print("Can't receive video")
        break

    diff = cv2.absdiff(gray1, gray2)
    _, trash = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(trash, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt)>100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 0, 0), 2)

    gray1 = gray2


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("Gray", gray2)
    cv2.imshow("Video", frame2)


vid.release()
#Звільняю камеру від використання
cv2.destroyAllWindows()