import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

# --- 1. Define CVAE Model Architecture ---
# These classes must be defined in the Streamlit app file
# because torch.load_state_dict() requires the model class definition.
# Ensure these hyperparameters match those used during your model training.
latent_dim = 20
num_classes = 10
image_size = 28
image_channels = 1 # Grayscale MNIST images

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(Encoder, self).__init__()
        # Input to encoder: flattened image (784) + one-hot label (10)
        self.fc1 = nn.Linear(input_dim + num_classes, 400)
        self.fc21 = nn.Linear(400, latent_dim) # Mean (mu)
        self.fc22 = nn.Linear(400, latent_dim) # Log-variance (log_var)

    def forward(self, x, labels):
        # Flatten the image
        x = x.view(-1, image_size * image_size)
        
        # One-hot encode labels and concatenate with image input
        one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
        x = torch.cat([x, one_hot_labels], dim=1)

        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        log_var = self.fc22(h1)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_classes):
        super(Decoder, self).__init__()
        # Input to decoder: latent vector (latent_dim) + one-hot label (10)
        self.fc3 = nn.Linear(latent_dim + num_classes, 400)
        self.fc4 = nn.Linear(400, output_dim)

    def forward(self, z, labels):
        # One-hot encode labels and concatenate with latent vector
        one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
        z = torch.cat([z, one_hot_labels], dim=1)

        h3 = F.relu(self.fc3(z))
        # Output layer uses Sigmoid for pixel values between 0 and 1
        recon_x = torch.sigmoid(self.fc4(h3))
        return recon_x.view(-1, image_channels, image_size, image_size)

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, input_dim, num_classes)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        mu, log_var = self.encoder(x, labels)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z, labels)
        return recon_x, mu, log_var

    def generate(self, z, labels):
        # For inference, directly use the decoder with a sampled latent vector and label
        with torch.no_grad():
            generated_image = self.decoder(z, labels)
        return generated_image

# --- 2. Model Loading with Caching ---
# @st.cache_resource decorator ensures the model is loaded only once
# across all user sessions and reruns, optimizing performance.
@st.cache_resource
def load_cvae_model(model_path):
    # Streamlit Community Cloud typically runs on CPU, so force map_location to 'cpu'.
    device = torch.device('cpu') # [2]
    
    model = CVAE(input_dim=image_size * image_size, latent_dim=latent_dim, num_classes=num_classes).to(device)
    
    # Load the state_dict, ensuring it's mapped to the CPU. [3, 4]
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() # Set the model to evaluation mode for consistent inference results. [3, 4]
    return model

# Define the path to your trained model file.
# Ensure this file is present in the same directory as your app.py for deployment.
MODEL_PATH = 'cvae_mnist_epoch_50(1).pth' # Adjust if your file name is different

# Attempt to load the model. If not found, display an error and stop the app.
try:
    model = load_cvae_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the same directory as app.py.")
    st.stop() # Stop the app execution if the model cannot be loaded.

# --- 3. Streamlit User Interface ---
st.set_page_config(layout="centered", page_title="Handwritten Digit Image Generator")

st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

# Allow users to select a digit from 0 to 9.
selected_digit = st.selectbox(
    "Choose a digit to generate (0-9):",
    options=list(range(10)),
    index=2 # Default selection to '2' as per the example image.
)

# Button to trigger the image generation process.
if st.button("Generate Images"):
    st.subheader(f"Generated images of digit {selected_digit}")

    generated_images_list = []# Corrected: Initialize as an empty list
    captions_list = []# Corrected: Initialize as an empty list
    num_samples_to_generate = 5 # Generate 5 images as per the requirement. [User Query]

    # Generate multiple images for the selected digit.
    for i in range(num_samples_to_generate):
        # Sample a new random latent vector for each image to ensure diversity. [User Query]
        # Ensure the latent vector is on the same device as the model.
        z = torch.randn(1, latent_dim).to(model.decoder.fc3.weight.device)
        
        # Create a one-hot encoded tensor for the selected digit.
        target_label = torch.tensor([selected_digit]).to(model.decoder.fc3.weight.device)
        
        # Generate the image using the CVAE model.
        img_tensor = model.generate(z, target_label)
        
        # Convert the PyTorch tensor to a NumPy array and then to a PIL Image for display.
        # Squeeze removes singleton dimensions (e.g., batch and channel dimensions).
        img_np = img_tensor.squeeze().cpu().numpy()
        # Scale pixel values from 0-1 to 0-255 and convert to uint8 for PIL.
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        
        generated_images_list.append(img_pil)
        captions_list.append(f"Sample {i+1}")

    # Display the generated images side-by-side with captions.
    # st.image can take a list of images and a list of captions. [5, 6]
    st.image(generated_images_list, caption=captions_list, width=100) # Adjust width for optimal display.

st.markdown("---")
st.write("Note: The model aims to generate similar but diverse images. The quality and diversity depend on the training duration and model architecture.")