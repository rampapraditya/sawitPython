import tensorflow as tf

# Path ke model yang telah dilatih
model_save_path = '../models/palm_maturity_model.keras'
tflite_model_save_path = '../models/model.tflite'

# Memuat model TensorFlow yang telah dilatih
model = tf.keras.models.load_model(model_save_path)

# Membuat converter untuk mengonversi model ke format TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Mengonversi model ke format TFLite
tflite_model = converter.convert()

# Menyimpan model TFLite ke file
with open(tflite_model_save_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model berhasil dikonversi ke TFLite dan disimpan di: {tflite_model_save_path}")
