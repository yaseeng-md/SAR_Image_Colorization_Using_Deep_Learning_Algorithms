import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from utils.UNetGenerator import UNetGenerator

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
generator = UNetGenerator(in_channels=3, out_channels=3).to(device)
generator.load_state_dict(torch.load("Models/Pix2Pix/GEN/GEN.pth", map_location=device))
generator.eval()

def process_and_colorize(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((256, 256), Image.Resampling.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = generator(input_tensor)

    output_tensor = (output_tensor + 1) / 2.0
    output_tensor = output_tensor.clamp(0, 1)

    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
    return output_image

# Streamlit UI
st.title("Grayscale to Color Image Colorization")

uploaded_file = st.file_uploader("Upload a grayscale or RGB image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Colorizing..."):
        output_image = process_and_colorize(uploaded_file)

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, width=300)

    with col2:
        st.subheader("Colorized Image")
        st.image(output_image, width=300)
