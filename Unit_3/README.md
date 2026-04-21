# Unit 3: Denoising Autoencoder & Anomaly Detection

Train a convolutional denoising autoencoder on FIDS30, then use reconstruction error for anomaly detection against satellite imagery.

## Notebooks

* **`00_PrepData.ipynb`**: Downloads FIDS30 and UC Merced, prepares train/val/test splits.
* **`01_DenoisingAutoencoder.ipynb`**: Trains a Conv autoencoder to denoise fruit images (128×128).
* **`02_AnomalyDetection.ipynb`**: Uses reconstruction error to detect satellite images as anomalies. ROC/AUC evaluation.

## Usage

1. Run `uv sync` to install dependencies.
2. Run notebooks sequentially (00 → 01 → 02).
