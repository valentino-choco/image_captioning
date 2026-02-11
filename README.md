# Deep Visual-Semantic Alignments for Image Captioning

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

video (in french) : https://youtu.be/xo1HeVU_LW0?si=cuM-JfsuDxccNLxy

This project is an educational implementation and modernization of the seminal paper **"Deep Visual-Semantic Alignments for Generating Image Descriptions"** by Andrej Karpathy and Fei-Fei Li (CVPR 2015).

The goal is to build a Deep Learning model capable of "looking" at an image and generating a coherent textual description in natural language. The architecture is split into two main components: **Visual-Semantic Alignment** (mapping image regions to words) and **Caption Generation** (using RNNs).

## 📄 Project Overview

This repository explores the connection between Computer Vision and Natural Language Processing (NLP) through the **Flickr30k** dataset.

The project is divided into several stages:
1.  **Data Analysis:** Understanding the dataset structure.
2.  **Alignment (The "Eye"):** Training a model to map visual features (R-CNN/ResNet) and text embeddings (BRNN) into a common space using **Contrastive Ranking Loss**.
    * *Note:* A modern variation using **OpenAI's CLIP** is also implemented for superior performance.
3.  **Captioning (The "Mouth"):** Training an Encoder-Decoder model (LSTM) where visual features are injected at every time step to generate sentences.
4.  **Evaluation:** Computing metrics like **CIDEr**, BLEU, and METEOR.

## 📂 Project Structure

The project is organized into sequential notebooks to facilitate understanding and reproducibility:

| File / Folder | Description |
| :--- | :--- |
| **`0_presentation_data.ipynb`** | **Data Exploration.** Presentation of the Flickr30k dataset (31,000+ images, 5 captions/image), vocabulary building, and data loading pipeline. |
| **`1_alignment.ipynb`** | **Alignment (From Scratch).** Implementation of the Visual-Semantic Alignment model using a pre-trained **Faster R-CNN** for region proposals and a **Bidirectional LSTM** for text, trained with Hinge Loss. |
| **`1_bis_alignment_clip.ipynb`** | **Alignment (Modernized).** An alternative implementation using **CLIP (Contrastive Language-Image Pre-Training)** to extract high-quality visual and textual features, significantly improving alignment stability. |
| **`2_captioning.ipynb`** | **Captioning Model.** Implementation of the Encoder-Decoder architecture. The decoder is an **LSTM** that receives the visual vector (from Part 1) concatenated with word embeddings at each time step. |
| **`3_visualize_results.ipynb`** | **Qualitative Evaluation.** Visualization of the generated captions on test images. Includes "Temperature" sampling analysis to observe the trade-off between precision and creativity. |
| **`4_scores.ipynb`** | **Quantitative Evaluation.** Calculation of standard NLP metrics: **BLEU**, **METEOR**, and **CIDEr** to rigorously assess the quality of the generated descriptions. |
| **`5_extension_webcam.ipynb`** | **Live Demo.** A real-time extension using a webcam to caption the video feed on the fly. |
| `cached_features/` | **Cache Directory.** Stores pre-computed features (`.pt` files) extracted from images (via ResNet/CLIP) to drastically speed up training and avoid redundant computations. |
| `saved_models/` | **Checkpoints.** Directory where trained model weights (`.pth` files) are saved during training. |

## 🛠️ Installation & Requirements

To run this project, you will need **Python 3.8+**.

First, install all the necessary dependencies (including download tools) using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Dataset Setup:

Since the dataset is large (2.5 GB), it is not stored directly in this repository. You need to download it before running the project.

- Download the data: Run the included script to automatically fetch the zip file:

```bash
python download_data.py
```

- Unzip the file: Extract the content of `cached_features.zip` into the `cached_features/` directory:

```bash
# On Linux/Mac
unzip cached_features.zip -d cached_features/

# On Windows (PowerShell)
Expand-Archive cached_features.zip -DestinationPath cached_features/
```

You are now ready to run the project! 🚀
