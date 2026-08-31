# Handwritten Digit Image Generator

A Streamlit-based web application that generates handwritten digit images using a **Conditional Variational Autoencoder (CVAE)** trained on the MNIST dataset.

## Overview

This project implements a deep learning model that can generate realistic handwritten digit images (0-9) conditioned on the desired digit class. The model is trained using a CVAE architecture, which combines the power of variational autoencoders with class conditioning to produce diverse and accurate digit samples.

## Features

- **Generate custom digits**: Specify which digit (0-9) you want to generate
- **Variational generation**: Each run produces unique variations of the requested digit
- **Interactive web interface**: Built with Streamlit for easy access and visualization
- **Pre-trained model**: Includes a model trained for 50 epochs on MNIST data
- **CPU-optimized**: Works efficiently on standard CPU environments

## Project Structure

```
Handwritten Digit Image Generator/
├── main.py                          # Streamlit application
├── cvae_mnist_epoch_50(1).pth       # Pre-trained CVAE model
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Requirements

- Python 3.8+
- PyTorch
- Streamlit
- NumPy
- Pillow (PIL)

## Installation

1. Clone or download this project
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run main.py
```

The application will open in your web browser. Use the interface to:

1. Select the digit class (0-9) you want to generate
2. Adjust sampling parameters if available
3. Generate one or more digit images
4. Download or save the generated images

## Model Architecture

### CVAE Components

**Encoder**:

- Input: Flattened MNIST image (784 values) + one-hot encoded label (10 values)
- Output: Mean (μ) and log-variance (log σ²) of latent distribution
- Architecture: 794 → 400 → 20 (latent_dim)

**Decoder**:

- Input: Latent vector (20 dimensions) + one-hot encoded label (10 values)
- Output: Reconstructed image (28×28 grayscale)
- Architecture: 30 → 400 → 784

**Latent Dimension**: 20
**Number of Classes**: 10 (digits 0-9)
**Image Size**: 28×28 pixels
**Channels**: 1 (grayscale)

## How It Works

1. **Class Conditioning**: The desired digit class is one-hot encoded and concatenated with the input/latent vectors
2. **Variational Sampling**: The latent space is sampled from a learned distribution, enabling diverse generation
3. **Reparameterization Trick**: Enables backpropagation through the stochastic sampling layer during training
4. **Inference Mode**: The decoder generates new images by sampling random latent vectors conditioned on the selected digit class

## Dependencies

| Package     | Purpose                   |
| ----------- | ------------------------- |
| streamlit   | Web application framework |
| torch       | Deep learning framework   |
| torchvision | Computer vision utilities |
| numpy       | Numerical computing       |
| Pillow      | Image processing          |

## Performance

- **Training**: Trained for 50 epochs on MNIST dataset
- **Runtime**: Optimized for CPU inference
- **Generation Speed**: Typically generates images in <1 second
