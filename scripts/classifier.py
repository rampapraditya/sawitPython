import numpy as np
from keras.preprocessing import image


class Classifier:

    def __init__(self, model, img_size=(150, 150)):
        self.model = model
        self.img_size = img_size

    def predict(self, img_path):
        # Proses gambar
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Lakukan prediksi
        predictions = self.model.predict(img_array)
        return np.argmax(predictions)
