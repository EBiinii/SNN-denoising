# SNN-denoising

# Layer-wise Excitatory-Inhibitory Spiking Neural Networks for Structure-Aware Image Denoising

This repository provides the official implementation of **Layer-wise Excitatory-Inhibitory Spiking Neural Networks for Structure-Aware Image Denoising**.

The proposed framework combines the spatial feature extraction capability of a **ResNet-18 backbone** with the temporal processing characteristics of **Spiking Neural Networks (SNNs)**. Excitatory and inhibitory spiking neurons are incorporated at different locations in the network to investigate their effects on image restoration performance.

The repository contains the training, evaluation, and repeated-experiment scripts used in the study.

---

## Overview

Deep learning-based image denoising methods have achieved significant improvements in image restoration performance. However, conventional CNN-based models rely on continuous-valued operations and may require substantial computational resources.

In this study, we investigate an SNN-based image denoising framework that combines the spatial feature extraction capability of ResNet-18 with spike-based temporal processing.

The proposed architecture incorporates:

- **ResNet-18** for hierarchical spatial feature extraction
- **Leaky Integrate-and-Fire (LIF) neurons** for spike-based processing
- **Excitatory neurons** at different locations in the network
- **Excitatory-inhibitory (E-I) neuron pairs** for regulating spike activity
- **Layer-wise SNN integration** for structure-aware image restoration

The proposed model was evaluated under various image corruption conditions, including Gaussian noise, shot noise, horizontal stripe noise, vertical stripe noise, and combined corruption environments.

---

## Key Features

- ResNet-18-based image denoising architecture
- Layer-wise integration of SNN components
- LIF spiking neurons implemented using `snnTorch`
- Excitatory and inhibitory neuron interactions
- Five-step temporal spike processing
- Multiple noise and corruption conditions
- PSNR and SSIM-based evaluation
- Five independent repeated experiments
- Comparison with conventional and SNN-based denoising methods

---

## Methodology

The proposed framework combines conventional deep feature extraction with spike-based temporal processing.

### ResNet-18 Backbone

A ResNet-18 backbone is used to extract hierarchical spatial features from noisy images.

The residual blocks progressively extract spatial features while residual connections help preserve information across the network.

### Spiking Neurons

LIF neurons are incorporated into the network to process feature representations in the spike domain.

The main SNN settings are:

| Parameter | Setting |
|---|---:|
| Neuron model | LIF |
| Number of time steps | 5 |
| Membrane decay (`beta`) | 0.5 |
| Surrogate gradient | Fast Sigmoid |
| Surrogate gradient slope | 25 |

The network processes feature representations over five temporal steps.

### Excitatory-Inhibitory Neurons

Excitatory neurons activate and transmit feature information, while inhibitory neurons suppress excessive or unnecessary spike activity.

An excitatory-to-inhibitory neuron ratio of **4:1** was used as a biologically motivated design choice.

The 4:1 ratio was not empirically optimized in this study and is not claimed to be the optimal ratio for image denoising.

### Layer-wise Neuron Placement

Different SNN neuron placement configurations were investigated to examine how the location of spiking neurons affects image restoration.

The experiments include:

- Excitatory neurons at selected layers
- Excitatory neurons after ResNet-18 blocks
- Excitatory-inhibitory neuron pairs
- Layer-wise excitatory and inhibitory neuron placement

These experiments investigate the interaction between hierarchical spatial feature extraction and spike-based processing.

---

## Noise and Corruption Conditions

The proposed model was evaluated under several types of image corruption.

### Gaussian Noise

Gaussian noise was evaluated at the following noise intensities:

- σ = 5
- σ = 10
- σ = 15

Each Gaussian noise level was trained and evaluated separately.

Therefore, these experiments evaluate the model's performance under different Gaussian noise intensities and do not represent unseen-noise-level generalization.

### Shot Noise

The model was evaluated under shot noise to investigate its ability to restore images affected by intensity-dependent noise.

### Horizontal Stripe Noise

Horizontal stripe noise was used to evaluate the restoration of directional sensor-related corruption.

### Vertical Stripe Noise

Vertical stripe noise was separately evaluated to examine the model's response to another directional corruption pattern.

### Combined Corruptions

The proposed model was also evaluated under combined corruption environments:

- Vertical stripe + shot noise + Gaussian noise
- Vertical stripe + Gaussian noise
- Vertical stripe + shot noise
- Horizontal stripe + shot noise + Gaussian noise
- Horizontal stripe + Gaussian noise
- Horizontal stripe + shot noise

These experiments were designed to investigate the robustness of the proposed architecture under complex corruption conditions.

---

## Datasets

The experiments were conducted using publicly available image datasets.

### Set12

Set12 is a commonly used grayscale image benchmark for image denoising evaluation.

### Urban100

Urban100 contains urban and architectural images with detailed structures and repetitive textures.

### BSD

The Berkeley Segmentation Dataset (BSD) was used to evaluate the denoising performance on natural images with diverse structures and textures.

### ISIC 2018

The ISIC 2018 dataset was used to evaluate the proposed method on skin lesion images.

Official dataset website:

https://challenge.isic-archive.com/data/

Please download the datasets from their official sources and configure the dataset paths according to the local environment.

---

## Repository Structure

```text
Layer-wise-EI-SNN-Denoising/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── train_s.py
├── test_s.py
├── test_5.py
│
├── models.py
├── models_comparison.py
├── utils.py
│
└── ...
