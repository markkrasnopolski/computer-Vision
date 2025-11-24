import keras
import pandas as pd
import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib as plt

# читаємо csv
df = pd.read_csv('data/figures.csv')
print(df)

# перетворюємо значення ст. label з csv
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

# обираємо елементи для навчання
X = df[['area', 'perimetr', 'corners']]
y = df['label_enc']

# створюємо модель
model = keras.Sequential([layers.Dense(8, activation = "relu", input_shape = (3,)),
                          layers.Dense(8, activation = "relu"),
                          layers.Dense(8, activation = "softmax")])
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentrophy',
    metrics = ['accuracy']
)

# навчання моделі
history = model.fit(X, y, epochs = 200, verbose = 0)

# візуалізація навчання
plt.plot(history.history['loss'], label = 'Втрата (Loss)')
plt.plot(history.history['accuracy'], label = 'Точність (Accuracy)')
plt.xlabel('Епоха')
plt.ylabel('Значення')
plt.title('Процес навчання')
plt.legend()
plt.show()

# тестування
test = np.array([18, 16, 0])
pred = model.predict(test)
print(f'Імовірність по кожному класу: {pred}')
print(f'Модель визначила: {encoder.inverse_transform([np.argmax(pred)])}')