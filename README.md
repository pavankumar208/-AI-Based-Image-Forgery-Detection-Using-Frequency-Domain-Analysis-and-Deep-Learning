
# AI-Based Image Forgery Detection Using Frequency Domain Analysis and Deep Learning

This project implements a simple and efficient image forgery detection system using frequency domain analysis (FFT) and a Convolutional Neural Network (CNN). The model is trained on synthetically generated image data to distinguish between real and forged images.

## Features

- Converts images to frequency domain using Fast Fourier Transform (FFT)
- Uses a custom CNN model for binary classification (Real vs Forged)
- Allows synthetic data generation for training
- Accepts user-uploaded images for prediction
- Visualizes FFT spectrum and prediction results

## Technologies Used

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Google Colab (optional for browser-based file uploads)

## Model

A lightweight CNN processes grayscale FFT images with the following architecture:

- 2 Convolutional layers with ReLU and MaxPooling
- Flatten + Fully Connected Layers
- Binary Classification Output (Real or Forged)

## How to Run

1. Install dependencies:
   ```bash
   pip install numpy opencv-python torch matplotlib
   ```

2. Run the script:
   ```bash
   python "AI-Based Image Forgery Detection Using Frequency Domain Analysis and Deep Learning.py"
   ```

3. Use the file upload prompt (if using Colab) or modify the script to load local images.

## Example Usage

- Generates synthetic training data by creating fake and real images
- Trains and saves a model (`fft_model.pth`)
- Accepts a new image for classification and displays results

## Output

- Console shows prediction results (e.g., "Real" or "Forged")
- Matplotlib plots the original image and its FFT spectrum

## License

This project is provided under the MIT License.
