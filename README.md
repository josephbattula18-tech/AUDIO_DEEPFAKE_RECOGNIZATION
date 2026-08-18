# 🎙️ AI vs. Human Voice Detection via Binary Embeddings

A deep learning project focused on identifying synthetic (AI-generated) voice clones and deepfake audio versus authentic human speech by extracting acoustic features and mapping them into high-density **binary voice embeddings**.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture & Approach](#-architecture--approach)
- [Tech Stack](#-tech-stack)
- [Directory Structure](#-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Results & Metrics](#-results--metrics)
- [Future Enhancements](#-future-enhancements)

---

## 📌 Overview

With the rapid advancement of neural text-to-speech (TTS) and voice cloning models (e.g., ElevenLabs, Bark, Tacotron 2), synthetic audio has become virtually indistinguishable from human speech to the human ear. 

This repository implements a robust binary classification pipeline designed to detect subtle acoustic artifacts, spectral anomalies, and phase inconsistencies present in AI-generated voices using **binary voice embeddings**.

---

## 🛠️ Key Features

- **Audio Preprocessing:** Automated pipeline to handle audio trimming, noise reduction, and normalization across varying sample rates.
- **Feature Extraction:** Extracts spectral properties including Mel-Frequency Cepstral Coefficients (MFCCs), Mel-Spectrograms, and Chroma features using `librosa`.
- **Binary Embedding Mapping:** Generates compact binary vector representations from speech signals to isolate high-frequency synthetic artifacts.
- **Deep Learning Classifier:** Uses custom neural architectures (CNN / PyTorch Model) optimized for low latency and high accuracy.
- **Model Evaluation:** Full diagnostic tools including confusion matrices, ROC-AUC curves, and Precision/Recall reports.

---

## 🧬 Architecture & Approach

1. **Input Audio:** Raw audio files (`.wav`, `.mp3`).
2. **Signal Processing:** Short-Time Fourier Transform (STFT) conversion to derive spectral representations.
3. **Embedding Layer:** Audio signals pass through an encoder network to map latent acoustic traits into binary embeddings.
4. **Classification:** A dense neural network evaluates the binary embeddings to output a probability score:
   - `0`: Authentic Human Speech
   - `1`: AI-Generated / Synthetic Speech

---

## 💻 Tech Stack

- **Primary Language:** Python 3.8+
- **Audio Processing:** Librosa, Torchaudio, SciPy, SoundFile
- **Machine Learning / Deep Learning:** PyTorch / TensorFlow, Scikit-Learn
- **Data Handling & Visualization:** NumPy, Pandas, Matplotlib, Seaborn

---

## 📁 Directory Structure

```text
├── assets/                  # Diagrams, confusion matrix plots, ROC curves
├── data/                    # Dataset directory (git-ignored)
│   ├── raw/                 # Raw human & AI audio samples
│   └── processed/           # Processed features and embeddings
├── models/                  # Saved model weights (.pt / .h5)
├── notebooks/               # Jupyter notebooks for EDA and experiment tracking
├── src/                     # Source code
│   ├── preprocess.py        # Audio loading, cleaning, and normalization
│   ├── feature_extraction.py# MFCCs & binary embedding generation logic
│   ├── train.py             # Model training and hyperparameter tuning loop
│   └── evaluate.py          # Metric generation and test set evaluation
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt         # Project dependencies
