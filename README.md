# Eye State Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify whether eyes are open or closed using the MRL Eye Dataset. The model is built using TensorFlow/Keras and includes data preprocessing, training, evaluation, and inference capabilities.

## Features

- **CNN Architecture**: Custom CNN with convolutional, pooling, and dense layers
- **Data Preprocessing**: Automatic image resizing, normalization, and augmentation
- **Training**: Adam optimizer with binary crossentropy loss
- **Evaluation**: Accuracy metrics and confusion matrix visualization
- **Inference**: Single image prediction and real-time webcam detection
- **Production Ready**: Clean, modular code with proper error handling

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Quick Start

### ⚠️ IMPORTANT: Python Version Requirement

**TensorFlow does not support Python 3.14.** You must use Python 3.11.

### Option 1: Automated Setup (Recommended)

1. **Run the setup script:**
   ```bash
   setup.bat
   ```
   This will guide you through installing Python 3.11 and all dependencies.

2. **Train the model:**
   ```bash
   train.bat
   ```

### Option 2: Manual Setup

1. **Install Python 3.11:**
   - Download from: https://www.python.org/downloads/release/python-3118/
   - Install and ensure it's added to PATH

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model:**
   ```bash
   python eye_classifier.py
   ```

### Option 3: Using Conda (if available)

```bash
conda create -n eye_classifier python=3.11
conda activate eye_classifier
pip install -r requirements.txt
python eye_classifier.py
```

   Or install manually:
   ```bash
   pip install tensorflow opencv-python numpy matplotlib scikit-learn pillow streamlit
   ```

## Run the Dashboard

After training the model and saving `eye_model.h5`, launch the dashboard with:

```bash
streamlit run app.py
```

Then open the browser page that Streamlit displays and upload an eye image.

## Dataset

This project is designed to work with the MRL Eye Dataset. The dataset should be organized as follows:

```
dataset/
├── open/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── closed/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

- **Open eyes**: Label = 1
- **Closed eyes**: Label = 0

Download the MRL Eye Dataset and place it in a directory accessible to the script.

## Usage

### Training the Model

1. Update the `data_dir` variable in `eye_classifier.py` with the path to your dataset:
   ```python
   data_dir = "path/to/your/dataset"
   ```

2. Run the training script:
   ```bash
   python eye_classifier.py
   ```

This will:
- Load and preprocess the data
- Train the CNN model for 20 epochs
- Display training/validation curves
- Show evaluation metrics and confusion matrix
- Save the trained model as `eye_model.h5`

### Single Image Prediction

Predict eye state for a single image:

```bash
python inference.py --image path/to/your/image.jpg
```

Example output:
```
Prediction: Eyes Open
Confidence: 87.34%
```

### Real-time Webcam Detection

Use your webcam for real-time eye detection:

```bash
python inference.py --webcam
```

This will:
- Open your webcam
- Detect eyes using Haar cascades
- Classify each detected eye as open/closed
- Display results in real-time
- Press 'q' to quit

## Model Architecture

The CNN architecture consists of:

- **Input**: 64x64 grayscale images
- **Conv2D Layer 1**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Dropout**: 25%
- **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Dropout**: 25%
- **Conv2D Layer 3**: 128 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Dropout**: 25%
- **Flatten**
- **Dense Layer**: 128 units, ReLU activation
- **Dropout**: 50%
- **Output Layer**: 1 unit, Sigmoid activation

## Training Parameters

- **Optimizer**: Adam (learning rate = 0.001)
- **Loss**: Binary Crossentropy
- **Batch Size**: 32
- **Epochs**: 20 (with early stopping)
- **Data Augmentation**: Rotation, zoom, horizontal flip

## Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Loss**: Binary crossentropy loss
- **Confusion Matrix**: True positives, false positives, etc.

## File Structure

```
├── eye_classifier.py      # Main training and evaluation script
├── inference.py           # Inference script for prediction
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── eye_model.h5          # Trained model (generated after training)
```

## Customization

### Model Architecture
Modify the `create_model()` method in `EyeClassifier` class to change the CNN architecture.

### Training Parameters
Adjust hyperparameters in the `train_model()` method:
- Learning rate
- Batch size
- Number of epochs
- Data augmentation parameters

### Image Size
Change the `img_size` parameter in the `EyeClassifier` constructor to use different input dimensions.

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Not Found**: Update the `data_dir` path in `eye_classifier.py`

3. **Webcam Not Working**: Ensure your camera is not being used by other applications

4. **Low Accuracy**: Try adjusting the model architecture or training parameters

### Performance Tips

- Use GPU acceleration if available (TensorFlow will automatically detect CUDA)
- Increase batch size if you have sufficient RAM
- Add more data augmentation techniques for better generalization
- Use transfer learning with pre-trained models for improved performance

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Citation

If you use this code in your research, please cite:

```
Eye State Classification using CNN
Author: ML Engineer
Date: April 2026
```