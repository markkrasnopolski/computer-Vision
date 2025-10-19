import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            object_detected = True
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                points.append((cx, cy))

    if object_detected:
        color = (0, 0, 255)
        text = "Object detected"
        text_color = (0, 255, 0)
    else:
        color = (0, 255, 0)
        text = "Object not detected"
        text_color = (0, 0, 255)

    cv2.rectangle(frame, (2, 2), (frame.shape[1] - 2, frame.shape[0] - 2), color, 3)
    cv2.putText(frame, text, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    cv2.imshow("Video", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
