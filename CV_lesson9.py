import cv2

net = cv2.dnn.readNetFromCaffe("data/MobileNet/mobilenet_deploy.prototxt", 'data/MobileNet/MobileNet.caffemodel')
#Завантажуємо модель

classes =[]
with open('data/MobileNet/synset.txt', "r", encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)
#Зчитаємо список класів моделі

image = cv2.imread('data/MobileNet/cat.jpg')

blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))
#Адаптуємо зображення під модель

net.setInput(blob)
# Кладемо в мереєжу підготовлені файли

preds = net.forward()
# Вектор імовірності для класів