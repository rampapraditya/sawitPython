from scripts.model import PalmMaturityModel
from scripts.classifier import Classifier


palm_model = PalmMaturityModel(input_shape=(150, 150, 3), num_classes=5)
classifier = Classifier(palm_model.load_model("models/palm_maturity_model.keras"))

# Prediksi pada gambar uji
test_image_path = 'C:/Users/Pc Mobile 08/Desktop/Ripe/1.png'  # Path ke gambar uji
predicted_class = classifier.predict(test_image_path)

# Menampilkan hasil prediksi
labels = ['Empty', 'OverRipe', 'Ripe', 'UnderRip', 'Unripe']
print(f"Prediksi Tingkat Kematangan: {labels[predicted_class]}")