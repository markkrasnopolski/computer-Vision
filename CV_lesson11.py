import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255)
}

X = []
y = []

for color_name, bgr in colors.items():
    for _ in range(200):
        noisy = np.clip(np.array(bgr) + np.random.randint(-20, 21, 3), 0, 255)
        X.append(noisy)
        y.append(color_name)

X = np.array(X)
y = np.array(y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

def detect_shape(contour):
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    corners = len(approx)
    shape = "unidentified"

    if corners == 3:
        shape = "triangle"
    elif corners == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.9 <= aspect_ratio <= 1.1:
            shape = "square"
        else:
            shape = "rectangle"
    elif corners > 5:
        shape = "circle"
    return shape

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0, 50, 50), (180, 255, 255))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_count = {}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = frame[y:y+h, x:x+w]

        mean_color = cv2.mean(roi)[:3]
        mean_color_bgr = np.array(mean_color).reshape(1, -1)

        color_label = model.predict(mean_color_bgr)[0]
        shape = detect_shape(cnt)

        color_count[color_label] = color_count.get(color_label, 0) + 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, f"{color_label} {shape}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    cv2.imshow("Color & Shape Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()