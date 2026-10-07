# SNN-denoising

This repository provides the implementation of **Layer-wise Excitatory-Inhibitory Spiking Neural Networks for Structure-Aware Image Denoising**.

The proposed framework combines the spatial feature extraction capability of a **ResNet-18 backbone** with the temporal processing characteristics of **Spiking Neural Networks (SNNs)**. Excitatory and inhibitory spiking neurons are incorporated at different locations in the network to investigate how spike-based processing and excitatory-inhibitory interactions affect image restoration performance.

The repository contains the network architecture, training and evaluation scripts, comparison models, and utility functions used in the experiments reported in the manuscript.

---

## Overview

Image denoising is an important preprocessing step for many computer vision and image analysis applications. Although deep learning-based denoising methods have achieved high restoration performance, conventional CNN architectures rely on continuous-valued operations and can require substantial computational resources.

To address this issue, this study investigates an SNN-based image denoising framework that combines conventional spatial feature extraction with spike-based processing.

The proposed architecture uses:

- A **ResNet-18 backbone** for hierarchical spatial feature extraction
- **Leaky Integrate-and-Fire (LIF) neurons** for spike-based processing
- **Excitatory neurons** placed at different network locations
- **Excitatory-inhibitory (E-I) neuron pairs** for regulating spike activity
- A layer-wise neuron placement strategy for structure-aware image restoration

The model was evaluated under several image corruption conditions, including Gaussian noise, shot noise, horizontal stripe noise, vertical stripe noise, and combinations of these corruptions.

---

## Key Features

- ResNet-18-based image denoising architecture
- Layer-wise integration of SNN components
- LIF spiking neurons implemented using `snnTorch`
- Excitatory and inhibitory neuron interactions
- Five-step temporal spike processing
- Evaluation under multiple noise types and intensities
- Comparison with conventional image denoising architectures
- Support for PSNR and SSIM evaluation
- Reproducible training and testing pipeline

---

## Methodology

The proposed framework combines conventional deep feature extraction with spike-based temporal processing.

### ResNet-18 Backbone

A ResNet-18 backbone is used to extract hierarchical spatial features from noisy images. Residual blocks provide progressively higher-level representations while maintaining information through residual connections.

### Spiking Neurons

LIF neurons are incorporated into the network to process feature representations in the spike domain.

The main SNN settings used in the experiments are:

| Parameter | Setting |
|---|---:|
| Neuron model | LIF |
| Number of time steps | 5 |
| Membrane decay (`beta`) | 0.5 |
| Surrogate gradient | Fast Sigmoid |
| Surrogate gradient slope | 25 |

The temporal dimension allows the network to process feature information over multiple spike-processing steps.

### Excitatory-Inhibitory Neurons

Both excitatory and inhibitory neurons are used to regulate spike activity.

Excitatory neurons transmit activated feature information, while inhibitory neurons suppress excessive or unnecessary spike activity. An excitatory-to-inhibitory neuron ratio of **4:1** was used as a biologically motivated design choice.

The 4:1 ratio was not selected through an optimization study and is not claimed to be the optimal ratio for image denoising.

### Layer-wise Neuron Placement

Different neuron placement configurations were investigated to determine how the location of SNN components affects denoising performance.

The experiments include configurations with:

- Excitatory neurons at selected layers
- Excitatory neurons after ResNet-18 blocks
- Excitatory-inhibitory neuron pairs
- Layer-wise excitatory and inhibitory neuron placement

These experiments were used to investigate the relationship between spatial feature extraction and spike-based processing.

---

## Noise and Corruption Conditions

The proposed model was evaluated under several types of image corruption.

### Gaussian Noise

Gaussian noise was evaluated at multiple noise intensities:

- σ = 5
- σ = 10
- σ = 15

For these experiments, each Gaussian noise level was trained and evaluated separately.

Therefore, these experiments evaluate performance under different Gaussian noise intensities rather than unseen-noise-level generalization.

### Shot Noise

The model was also evaluated under shot noise to investigate its ability to restore images affected by intensity-dependent noise.

### Horizontal Stripe Noise

Horizontal stripe artifacts were used to evaluate the model's ability to restore directional sensor-related corruption.

### Vertical Stripe Noise

Vertical stripe corruption was evaluated separately to examine the response to another directional image degradation pattern.

### Combined Corruptions

The model was further evaluated under combinations of different corruption types, including:

- Vertical stripe + shot noise + Gaussian noise
- Vertical stripe + Gaussian noise
- Vertical stripe + shot noise
- Horizontal stripe + shot noise + Gaussian noise
- Horizontal stripe + Gaussian noise
- Horizontal stripe + shot noise

These experiments were designed to evaluate model robustness under more complex corruption environments.

---

## Datasets

The experiments were conducted using publicly available image datasets.

### Set12

Set12 is a commonly used grayscale image benchmark for evaluating image denoising algorithms.

### Urban100

Urban100 contains urban and architectural images with detailed structural patterns and repetitive textures.

### BSD

The Berkeley Segmentation Dataset (BSD) was used to evaluate denoising performance on natural images with diverse structures and textures.

### ISIC 2018

The ISIC 2018 dataset was used to evaluate the proposed framework on skin lesion images.

Official dataset information:

https://challenge.isic-archive.com/data/

Please download the datasets from their respective official sources and organize them according to the paths expected by the data-loading scripts.

---

## Repository Structure

```text
Layer-wise-EI-SNN-Denoising/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── train.py
├── test.py
├── test_reviewer.py
│
├── models.py
├── models_comparison.py
├── utils.py
│
├── dataset/
│   └── ...
│
├── checkpoints/
│   └── ...
│
└── results/
    └── ...
