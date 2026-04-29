"""
Eye State Classification using CNN
===================================

This project implements a Convolutional Neural Network (CNN) to classify
whether eyes are open or closed using the MRL Eye Dataset.

Requirements:
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

Author: ML Engineer
Date: April 2026
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


class EyeClassifier:
    """
    CNN-based classifier for eye state detection (open/closed).
    """

    def __init__(self, img_size=(64, 64), batch_size=32):
        """
        Initialize the eye classifier.

        Args:
            img_size (tuple): Target image size (height, width)
            batch_size (int): Batch size for training
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None

    def create_model(self):
        """
        Create the CNN model architecture.

        Returns:
            tf.keras.Model: Compiled CNN model
        """
        model = Sequential([
            # Convolutional layers
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 1)),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Flatten and dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification
        ])

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def load_and_preprocess_data(self, data_dir, validation_split=0.2):
        """
        Load and preprocess the dataset.

        Args:
            data_dir (str): Path to dataset directory
            validation_split (float): Fraction of data for validation

        Returns:
            tuple: (train_generator, validation_generator, train_samples, val_samples)
        """
        # For this specific dataset structure, we'll use the train folder
        # and create our own validation split from it
        train_data_dir = os.path.join(data_dir, "dataset_new", "train")

        # Create temporary directories for binary classification
        temp_dir = os.path.join(data_dir, "temp_binary_dataset")
        temp_train_dir = os.path.join(temp_dir, "train")
        temp_val_dir = os.path.join(temp_dir, "validation")

        # Clean up any existing temp directory
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)

        # Create temp directories
        os.makedirs(os.path.join(temp_train_dir, "open"), exist_ok=True)
        os.makedirs(os.path.join(temp_train_dir, "closed"), exist_ok=True)
        os.makedirs(os.path.join(temp_val_dir, "open"), exist_ok=True)
        os.makedirs(os.path.join(temp_val_dir, "closed"), exist_ok=True)

        # Copy and organize files for binary classification
        self._organize_binary_dataset(train_data_dir, temp_train_dir, temp_val_dir, validation_split)

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)

        # Load training data
        train_generator = train_datagen.flow_from_directory(
            temp_train_dir,
            target_size=self.img_size,
            color_mode='grayscale',
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True
        )

        # Load validation data
        validation_generator = val_datagen.flow_from_directory(
            temp_val_dir,
            target_size=self.img_size,
            color_mode='grayscale',
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )

        return train_generator, validation_generator

    def _organize_binary_dataset(self, source_dir, train_dest, val_dest, val_split):
        """
        Organize the dataset into binary classification format.

        Args:
            source_dir (str): Source dataset directory
            train_dest (str): Training destination directory
            val_dest (str): Validation destination directory
            val_split (float): Validation split ratio
        """
        import shutil

        # Map source classes to binary classes
        class_mapping = {
            'Open': 'open',      # eyes open
            'Closed': 'closed'   # eyes closed
        }

        for source_class, binary_class in class_mapping.items():
            source_class_dir = os.path.join(source_dir, source_class)

            if not os.path.exists(source_class_dir):
                print(f"Warning: {source_class_dir} not found")
                continue

            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(Path(source_class_dir).glob(ext))

            # Shuffle files
            np.random.shuffle(image_files)

            # Split into train/validation
            split_idx = int(len(image_files) * (1 - val_split))
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]

            # Copy training files
            for img_file in train_files:
                dest_path = os.path.join(train_dest, binary_class, img_file.name)
                shutil.copy2(str(img_file), dest_path)

            # Copy validation files
            for img_file in val_files:
                dest_path = os.path.join(val_dest, binary_class, img_file.name)
                shutil.copy2(str(img_file), dest_path)

            print(f"{source_class} -> {binary_class}: {len(train_files)} train, {len(val_files)} val")

    def train_model(self, train_generator, validation_generator, epochs=20):
        """
        Train the CNN model.

        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs (int): Number of training epochs

        Returns:
            tf.keras.callbacks.History: Training history
        """
        if self.model is None:
            self.create_model()

        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[early_stopping]
        )

        self.history = history
        return history

    def evaluate_model(self, validation_generator):
        """
        Evaluate the model on validation data.

        Args:
            validation_generator: Validation data generator

        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        # Get predictions
        val_predictions = self.model.predict(validation_generator)
        val_predictions_binary = (val_predictions > 0.5).astype(int).flatten()

        # Get true labels
        val_labels = validation_generator.classes

        # Calculate confusion matrix
        cm = confusion_matrix(val_labels, val_predictions_binary)

        # Get evaluation metrics
        loss, accuracy = self.model.evaluate(validation_generator, verbose=0)

        return {
            'accuracy': accuracy,
            'loss': loss,
            'confusion_matrix': cm,
            'predictions': val_predictions_binary,
            'true_labels': val_labels
        }

    def save_model(self, filepath='eye_model.h5'):
        """
        Save the trained model.

        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='eye_model.h5'):
        """
        Load a trained model.

        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def predict_image(self, image_path):
        """
        Predict eye state for a single image.

        Args:
            image_path (str): Path to the input image

        Returns:
            tuple: (prediction_label, confidence_score)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Load and preprocess image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Resize image
        image = cv2.resize(image, self.img_size)

        # Normalize
        image = image.astype('float32') / 255.0

        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=[0, -1])

        # Make prediction
        prediction = self.model.predict(image)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Determine label
        label = "Eyes Open" if prediction > 0.5 else "Eyes Closed"

        return label, confidence

    def plot_training_history(self):
        """
        Plot training and validation accuracy/loss curves.
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix.

        Args:
            cm (numpy.ndarray): Confusion matrix
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        classes = ['Closed', 'Open']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


def main():
    """
    Main function to run the complete pipeline.
    """
    # Initialize classifier
    classifier = EyeClassifier()

    # Dataset path - using the archive folder
    data_dir = "archive"  # Path to the archive folder containing dataset_new

    if not os.path.exists(data_dir):
        print(f"Dataset directory not found: {data_dir}")
        print("Please ensure the archive folder with dataset_new is in the current directory.")
        return

    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_generator, validation_generator = classifier.load_and_preprocess_data(data_dir)

    # Create and train model
    print("Creating model...")
    classifier.create_model()

    print("Training model...")
    history = classifier.train_model(train_generator, validation_generator, epochs=20)

    # Evaluate model
    print("Evaluating model...")
    eval_results = classifier.evaluate_model(validation_generator)

    print(".4f")
    print(f"Validation Loss: {eval_results['loss']:.4f}")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(eval_results['confusion_matrix'])

    # Save model
    classifier.save_model('eye_model.h5')

    # Plot results
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(eval_results['confusion_matrix'])

    print("\nTraining completed! Model saved as 'eye_model.h5'")


if __name__ == "__main__":
    main()