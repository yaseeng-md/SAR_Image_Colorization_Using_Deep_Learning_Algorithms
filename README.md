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




---

## ⚙️ Training Setup

- **Framework:** PyTorch
- **GPU:** NVIDIA DGX A100 (200GB)
- **Optimizer:** Adam
  - Generator: LR = 0.0002, β1 = 0.005, β2 = 0.99
  - Autoencoder: LR = 0.001
- **Input Shape (Generator):** (1, 256, 256, 1)
- **Input Shape (Discriminator):** (1, 256, 256, 2)

---

## 📊 Results

| Model                          | SSIM   | PSNR (dB) | MSE     |
|-------------------------------|--------|-----------|---------|
| Pix2Pix                       | 0.159  | 11.32     | -       |
| cGAN + SSIM + L1              | 0.353  | 16.28     | -       |
| Cycle GAN                     | 0.252  | 13.23     | -       |
| EPC-GAN                       | 0.188  | 12.07     | 0.0047  |
| **Pix2Pix + Perceptual Loss** | **0.97** | **27.42** | **0.0021** |

---

## 🚫 Limitations

- Limited training data (only 1000 images used).
- Performance dips with noisy or high-edge images.
- Currently focused on agricultural surfaces only.
- Edge and texture preservation still has room for improvement.

---

## 🛠️ Installation

```bash
git clone https://github.com/yaseeng-md/SAR_Image_Colorization_Using_Deep_Learning_Algorithms.git
cd SAR_Image_Colorization_Using_Deep_Learning_Algorithms
pip install -r requirements.txt

