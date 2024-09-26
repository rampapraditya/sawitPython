from keras.preprocessing.image import ImageDataGenerator

class DataLoader:

    def __init__(self, dataset_dir, img_size=(150, 150), batch_size=32):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Add validation split

    def load_train_data(self):
        # Generator untuk data pelatihan
        train_data = self.datagen.flow_from_directory(
            self.dataset_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'  # Menggunakan subset 'training'
        )
        return train_data

    def load_val_data(self):
        # Generator untuk data validasi
        val_data = self.datagen.flow_from_directory(
            self.dataset_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'  # Menggunakan subset 'validation'
        )
        return val_data
