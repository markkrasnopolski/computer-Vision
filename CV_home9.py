import cv2

net = cv2.dnn.readNetFromCaffe(
    'data/MobileNet/mobilenet_deploy.prototxt',
    'data/MobileNet/mobilenet.caffemodel'
)

classes = []
with open('data/MobileNet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(' ', 1)
            name = parts[1] if len(parts) > 1 else parts[0]
            classes.append(name)

photos = [
    'images/MobileNet/image1.jpg',
    'images/MobileNet/image2.jpg',
    'images/MobileNet/image3.jpg',
    'images/MobileNet/image4.jpg',
    'images/MobileNet/image5.jpg'
]

all_labels = []

for p in photos:
    img = cv2.imread(p)
    if img is None:
        print("Не вдалося відкрити:", p)
        continue

    img = cv2.resize(img, (800, 600))
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (224,224)),
        1.0/127.5,
        (224,224),
        (127.5,127.5,127.5)
    )
    net.setInput(blob)
    preds = net.forward()
    idx = preds[0].argmax()
    conf = float(preds[0][idx]) * 100
    label = classes[idx] if idx < len(classes) else "невідомо"
    all_labels.append(label)
    text = f"{label} ({conf:.1f}%)"
    cv2.putText(img, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    print(p.split('/')[-1], "→", label, f"({conf:.2f}%)")
    cv2.imshow("Результат", img)
    cv2.waitKey(700)

cv2.destroyAllWindows()

print("\nПідсумкова таблиця:")
unique = []
for lbl in all_labels:
    if lbl not in unique:
        unique.append(lbl)

for lbl in unique:
    count = 0
    for x in all_labels:
        if x == lbl:
            count += 1
    print(lbl, "-", count)
