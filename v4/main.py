import tensorflow as tf
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array

# Load model yang telah dilatih untuk klasifikasi kematangan kelapa sawit
model = tf.keras.models.load_model('../models/palm_maturity_model.keras')

# Kelas untuk tingkat kematangan kelapa sawit
class_indices = {0: 'Empty Bunch', 1: 'OverRipe', 2: 'Ripe', 3: 'UnderRip', 4: 'Unripe'}


# Fungsi untuk mendeteksi dan mengklasifikasi gambar
def predict_maturity(roi, model):
    img = cv2.resize(roi, (150, 150))  # Resize sesuai dengan ukuran input model
    img = img.astype("float") / 255.0  # Normalisasi
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Ubah menjadi 4D tensor

    # Prediksi menggunakan model
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    return class_indices[predicted_class], np.max(predictions)  # Return kelas dan probabilitas


# Load pre-trained SSD MobileNet dari OpenCV untuk deteksi objek
ssd_net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',  # File prototxt SSD MobileNet
    'ssd_mobilenet.caffemodel'  # Model pre-trained SSD MobileNet
)

# Definisi skala input dan resolusi untuk SSD
input_size = (300, 300)
scale = 0.007843  # 1/127.5 untuk normalisasi
mean = 127.5

# Batas probabilitas untuk menentukan objek
confidence_threshold = 0.5

# Mulai stream kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Ambil dimensi frame
    (h, w) = frame.shape[:2]

    # Pre-process frame untuk SSD
    blob = cv2.dnn.blobFromImage(frame, scale, input_size, mean, swapRB=True, crop=False)
    ssd_net.setInput(blob)

    # Deteksi objek menggunakan SSD
    detections = ssd_net.forward()

    # Iterasi melalui setiap deteksi
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # Ambil indeks kelas dan koordinat bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Potong ROI dari frame untuk klasifikasi kematangan
            roi = frame[startY:endY, startX:endX]

            # Lakukan prediksi kematangan hanya jika ukuran objek cukup besar
            if roi.shape[0] > 50 and roi.shape[1] > 50:
                maturity, confidence_maturity = predict_maturity(roi, model)

                # Gambarkan bounding box dan label tingkat kematangan di frame
                label = f"{maturity}: {confidence_maturity * 100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan hasil frame
    cv2.imshow("Palm Fruit Maturity Detection", frame)

    # Tekan 'q' untuk keluar dari stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan semua jendela dan hentikan kamera
cap.release()
cv2.destroyAllWindows()
