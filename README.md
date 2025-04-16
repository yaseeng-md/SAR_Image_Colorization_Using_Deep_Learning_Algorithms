# 🌈 SAR Image Colorization Using Pix2Pix GAN

This repository contains the implementation of SAR Image Colorization using a Conditional GAN (Pix2Pix), developed as part of a research project at Lovely Professional University. The project aims to colorize grayscale Sentinel-1 SAR images using paired Sentinel-2 optical images.

---

## 🧠 Overview

**SAR (Synthetic Aperture Radar)** images are grayscale and captured regardless of weather conditions, making them ideal for remote sensing. However, their lack of color limits visual interpretation. This project leverages **Pix2Pix GANs** with U-Net-based generator and PatchGAN discriminator to colorize SAR images.

---

## 📁 Dataset

- **Dataset Name:** SEN1-2
- **Image Size:** 256 x 256
- **Total Images:** 64,000 (50% SAR, 50% Optical)
- **Classes:** Agricultural, Urban, Barren, Grassland
- **Paired Dataset:** Each grayscale SAR image has a corresponding color optical image.

---

## 🔍 Methodology

### ✅ Generator
- U-Net architecture with skip connections.
- Downsamples the input using convolution, upsamples using transposed convolution.

### ✅ Discriminator
- PatchGAN: Operates on N x N patches instead of full images.
- Focuses on local features, improves fine detail generation.

### ✅ GAN Type
- **Pix2Pix**: Conditional GAN for image-to-image translation.

---

## 🎯 Loss Functions

- **L1 Loss:** Pixel-wise absolute error.
- **Perceptual Loss:** Measures feature map difference using pretrained VGG-16.
- **Adversarial Loss:** Binary cross entropy loss to fool the discriminator.
- **MSE Loss:** Used in Denoising Autoencoder for additional refinement.

### Combined Loss:
