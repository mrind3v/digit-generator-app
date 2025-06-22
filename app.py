# app.py

import streamlit as st
import torch
import numpy as np
from models import Generator  # Import the Generator class from models.py

# --- CONFIGURATION ---
MODEL_PATH = "generator.pth"
LATENT_DIM = 100  # Should match the latent_dim used during training

# --- MODEL LOADING ---

# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model():
    """Load the trained Generator model."""
    model = Generator()
    # Load the saved state dictionary.
    # We use map_location='cpu' to ensure the model loads on the CPU,
    # which is necessary for deployment on most standard services.
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    # Set the model to evaluation mode. This is important for inference.
    model.eval()
    return model

generator = load_model()

# --- WEB APP UI ---

st.set_page_config(layout="wide", page_title="Handwritten Digit Generation")

st.title("Handwritten Digit Generation Web App 🤖")
st.write("Select a digit from the dropdown menu and click 'Generate' to create five new handwritten-style images of that digit using a Conditional GAN model.")

# Create two columns for the controls and the output
col1, col2 = st.columns([1, 4])

with col1:
    # Dropdown for digit selection
    selected_digit = st.selectbox("Select a digit (0-9)", list(range(10)))
    
    # Button to trigger generation
    generate_button = st.button("Generate Images", type="primary")

# --- IMAGE GENERATION LOGIC ---

with col2:
    if generate_button:
        st.subheader(f"Generated Images for Digit: {selected_digit}")
        
        # Create 5 columns to display images side-by-side
        image_columns = st.columns(5)
        
        for i in range(5):
            # Generate a random noise vector
            noise = torch.randn(1, LATENT_DIM)
            
            # Create the label tensor for the selected digit
            label = torch.LongTensor([selected_digit])
            
            # Generate an image
            with torch.no_grad(): # Turn off gradient calculation for inference
                generated_img_tensor = generator(noise, label)
            
            # Post-process the tensor to be a displayable image
            # 1. Detach from the computation graph
            # 2. Move to CPU
            # 3. Convert to NumPy array
            # 4. Denormalize from [-1, 1] to [0, 1]
            # 5. Squeeze to remove batch and channel dimensions (1, 1, 28, 28) -> (28, 28)
            img_np = generated_img_tensor.detach().cpu().numpy().squeeze()
            img_np = (img_np + 1) / 2 # Denormalize
            
            # Display the image in its column
            with image_columns[i]:
                st.image(img_np, caption=f"Image {i+1}", use_column_width=True)
    else:
        st.info("Select a digit and click 'Generate Images' to see the results.")