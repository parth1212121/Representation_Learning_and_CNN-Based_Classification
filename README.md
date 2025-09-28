# Representation Learning and CNN-Based Classification

This repository contains two major components developed as part of COL774 (Machine Learning) coursework:

1. **Bird Classification using CNNs**  
2. **Representation Learning with Variational Autoencoder (VAE) + Gaussian Mixture Model (GMM)**

---

## 📖 Project Components

### 🐦 Bird Classification with CNN
- **Model**: Custom CNN with 5 convolutional blocks + fully connected layers (~1.7M parameters).  
- **Input**: Bird images resized to 300×200.  
- **Techniques Used**:
  - Data augmentation (horizontal/vertical flips, random rotations, color jitter).  
  - Regularization with Dropout and Batch Normalization.  
  - Custom **Warmup + Step Decay Learning Rate Scheduler**.  
- **Performance**:
  - Validation Accuracy: **97.29%**  
  - Macro F1 Score: **0.9711**  
  - Micro F1 Score: **0.9729**  
- **Interpretability**: Grad-CAM visualizations show that the model mainly focuses on **head, beak, and upper body regions** for classification.

Run:
```bash
# Training
python bird.py path_to_dataset train bird.pth

# Testing
python bird.py path_to_dataset test bird.pth
```
Predictions are saved in `bird.csv`.

---

### 🔢 Representation Learning: VAE + GMM
- **VAE**:
  - Convolutional Encoder–Decoder for digit image representation.  
  - Learns a low-dimensional latent space (2D).  
  - Generates new images and reconstructs input digits.  
- **GMM on Latent Space**:
  - Fits Gaussian Mixture Model (via EM algorithm) to cluster latent representations.  
  - Initialized with validation set class means (using 15 labelled samples).  
  - Provides classification in the latent space.

Run:
```bash
# Train VAE + GMM
python vae.py train_data.npz val_data.npz train vae.pth gmm.pkl

# Test reconstruction
python vae.py test_data.npz test_reconstruction vae.pth

# Test classification
python vae.py test_data.npz test_classifier vae.pth gmm.pkl
```
Outputs:
- `vae_reconstructed.npz` → reconstructed test images  
- `vae.csv` → classification predictions  

---

## 📂 Repository Structure
```
.
├── bird.py                     # CNN for bird classification
├── vae.py                      # VAE + GMM for representation learning
├── group.txt                   # Team member roll numbers
├── model_path.txt              # Links to pretrained models
├── 2022CS11936_2022CS11290.pdf # Bird classification report
├── 2022CS11936_2022CS11290_2.pdf # VAE + GMM report
└── README.md
```

---

## 📌 Results Summary
- **Bird CNN** achieved **97.29% accuracy**, robust to augmentation/regularization.  
- **VAE + GMM** produced meaningful clusters in latent space, with some overlap (esp. between digits 4 & 8).  

---

## 👥 Contributors
- Parth Verma (2022CS11936)  
- Aviral Singh (2022CS11290)  
