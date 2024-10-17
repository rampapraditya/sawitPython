import tensorflow as tf
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array

# Load model yang telah dilatih
model = tf.keras.models.load_model('../models/palm_maturity_model.keras')

# Kelas untuk tingkat kematangan kelapa sawit
class_indices = {0: 'Empty Bunch', 1: 'OverRipe', 2: 'Ripe', 3: 'UnderRip', 4: 'Unripe'}


# Fungsi untuk mendeteksi dan mengklasifikasi gambar
def predict_maturity(frame, model):
    img = cv2.resize(frame, (150, 150))  # Resize sesuai dengan ukuran input model
    img = img.astype("float") / 255.0  # Normalisasi
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Ubah menjadi 4D tensor

    # Prediksi menggunakan model
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    return class_indices[predicted_class], np.max(predictions)  # Return kelas dan probabilitas


# Fungsi untuk mendeteksi objek menggunakan bounding box
def detect_objects(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Thresholding untuk segmentasi objek
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Mencari kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


# Mulai stream kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Deteksi objek di frame
    contours = detect_objects(frame)

    # Iterasi melalui setiap kontur yang terdeteksi
    for contour in contours:
        # Gambarkan bounding box di sekitar kontur
        x, y, w, h = cv2.boundingRect(contour)

        # Ekstraksi objek dari frame
        roi = frame[y:y + h, x:x + w]

        # Lakukan prediksi hanya jika ukuran objek cukup besar
        if roi.shape[0] > 50 and roi.shape[1] > 50:
            maturity, confidence = predict_maturity(roi, model)

            # Gambarkan bounding box dan label tingkat kematangan di frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{maturity}: {confidence * 100:.2f}%"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan hasil frame
    cv2.imshow("Palm Fruit Maturity Detection", frame)

    # Tekan 'q' untuk keluar dari stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan semua jendela dan hentikan kamera
cap.release()
cv2.destroyAllWindows()
