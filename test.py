import cv2
import keras
import tensorflow as tf


def prepare(filepath):
    IMG_SIZE = 48
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


img_path = 'C:/MyProjects/pythonProject/input/test/fear/PrivateTest_166793.jpg'

x = tf.keras.Input(shape=(48, 48, 1))
y = tf.keras.layers.Dense(16, activation='softmax')(x)
model = tf.keras.Model(x, y)
model.summary()

model = keras.models.load_model('ferNet.h5')

preds = model.predict([prepare(img_path)])

print(preds)
