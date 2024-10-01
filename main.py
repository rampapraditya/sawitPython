from scripts.data_loader import DataLoader
from scripts.model import PalmMaturityModel
from scripts.classifier import Classifier

# Path dataset dan model
dataset_dir = 'dataset/'
model_save_path = 'models/palm_maturity_model.keras'

# Initialize DataLoader
data_loader = DataLoader(dataset_dir)

# Load data pelatihan dan validasi
train_data = data_loader.load_train_data()
val_data = data_loader.load_val_data()

# Inisialisasi model
palm_model = PalmMaturityModel(input_shape=(150, 150, 3), num_classes=5)

# Latih model
history = palm_model.train(train_data, val_data, epochs=150)

# Simpan model yang telah dilatih
palm_model.save_model(model_save_path)

# Inisialisasi classifier untuk prediksi
classifier = Classifier(palm_model.load_model(model_save_path))

# Prediksi pada gambar uji
test_image_path = 'dataset/Ripe/1.png'  # Path ke gambar uji
predicted_class = classifier.predict(test_image_path)

# Menampilkan hasil prediksi

labels = ['Empty', 'OverRipe', 'Ripe', 'UnderRip', 'Unripe']
print(f"Prediksi Tingkat Kematangan: {labels[predicted_class]}")