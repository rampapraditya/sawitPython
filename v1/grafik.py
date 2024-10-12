import matplotlib.pyplot as plt
from scripts.data_loader import DataLoader
from scripts.model import PalmMaturityModel
from scripts.classifier import Classifier

# Path dataset dan model
dataset_dir = '../dataset/'
model_save_path = '../models/palm_maturity_model.h5'

# Initialize DataLoader
data_loader = DataLoader(dataset_dir)

# Load data pelatihan dan validasi
train_data = data_loader.load_train_data()
val_data = data_loader.load_val_data()

# Inisialisasi model
palm_model = PalmMaturityModel(input_shape=(150, 150, 3), num_classes=5)

# Latih model
history = palm_model.train(train_data, val_data, epochs=150)


# Plot Akurasi dan Loss
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Plot Akurasi
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Tampilkan grafik akurasi dan loss
plot_training_history(history)