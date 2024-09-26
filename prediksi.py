from scripts.model import PalmMaturityModel
from scripts.classifier import Classifier


palm_model = PalmMaturityModel(input_shape=(150, 150, 3), num_classes=3)
classifier = Classifier(palm_model.load_model("models/palm_maturity_model1.keras"))

# Prediksi pada gambar uji
test_image_path = 'dataset/Unripe/23.png'  # Path ke gambar uji
predicted_class = classifier.predict(test_image_path)

# Menampilkan hasil prediksi
labels = ['Ripe', 'UnderRipe', 'Unripe']
print(f"Prediksi Tingkat Kematangan: {labels[predicted_class]}")