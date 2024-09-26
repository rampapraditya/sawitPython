import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import Sequential


class PalmMaturityModel:
    def __init__(self, input_shape=(150, 150, 3), num_classes=3):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    def train(self, train_data, val_data, epochs=10):
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs
        )
        return history

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        return self.model
