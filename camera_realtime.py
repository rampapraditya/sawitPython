import cv2
import numpy as np
from keras.preprocessing import image
from scripts.model import PalmMaturityModel
from scripts.classifier import Classifier

# Path ke model yang sudah dilatih
model_save_path = 'models/palm_maturity_model.keras'

# Label untuk kategori kematangan buah sawit
labels = ['Empty', 'OverRipe', 'Ripe', 'UnderRip', 'Unripe']

# Inisialisasi model yang telah dilatih
palm_model = PalmMaturityModel(input_shape=(150, 150, 3), num_classes=5)
model = palm_model.load_model(model_save_path)

# Inisialisasi classifier untuk prediksi
classifier = Classifier(model)

# Menggunakan kamera untuk menangkap gambar real-time
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera tidak dapat dibuka.")
    exit()

while True:
    # Tangkap frame dari kamera
    ret, frame = cap.read()

    if not ret:
        print("Gagal menangkap gambar.")
        break

    # Resize gambar untuk prediksi
    img_resized = cv2.resize(frame, (150, 150))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi menggunakan model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    label = labels[predicted_class]

    # Tampilkan hasil prediksi pada frame
    cv2.putText(frame, f'Maturity: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame dengan hasil prediksi
    cv2.imshow('Palm Maturity Detection', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan setelah penggunaan
cap.release()
cv2.destroyAllWindows()