# WavetableCVAE: From Labels to Waves

This repository presents a perceptually-driven approach to wavetable synthesis using a **Conditional Variational Autoencoder (CVAE)**. It enables intuitive timbre manipulation through high-level semantic descriptors.

## Project Overview

This work is an extension and modernization of the original research by [**Tsugumasa Yutani**](https://arxiv.org/pdf/2410.18628). While the core concept of generating single-cycle wavetables remains, this version introduces several key improvements in control dimensionality and technical stability.

### Original Foundation (by T. Yutani)
* **Generative Framework**: Introduction of the CVAE architecture for wavetable synthesis.
* **Semantic Labeling**: Conditioning the model on attributes like brightness, richness, and odd-energy.

### New Extensions & Contributions (this version)
* **Expanded Semantic Space**: Incorporated new morphological descriptors (Fullness, Undulation, and Symmetry) inspired by the [Wavespace framework](https://arxiv.org/pdf/2407.19862v1) to allow for a richer representation of timbre.
* **Model Modernization**: The implementation was upgraded to **PyTorch 2.2** and **PyTorch Lightning 2.2**, ensuring compatibility with modern CUDA-based GPUs and improving training stability.
* **Modular Conditioning Architecture**: Redesigned the conditioning layers using the **Hydra framework**, allowing the model to handle a dynamic number of semantic attributes without manual architecture changes.
* **Improved Signal Integrity**: Introduced a rigorous dataset cleaning procedure (DC offset removal, circular shift alignment, and edge ramping) to ensure smooth periodic boundaries and prevent audible clicks.
* **Interactive Evaluation**: Prototyped a web-based interface using **Gradio** for real-time manipulation and instant audio feedback of generated wavetables.


# Requirement


```
|
├── conf                   <- hydra config data
│
├── data                   <- Project data
│
├── src                    <- Source code
│   │
│   ├── check                    <- Visualization of generated results
│   ├── dataio                   <- Lightning datamodules
│   ├── models                   <- Lightning models
│   ├── tools                    <- utility tools
│   │
|   ├── utils.py                    <- Utility scripts
│   └── train.py                 <- Run training
│
├── torchscript            <- ckpt file
│
├── .gitignore                <- List of files ignored by git
├── requirements.txt          <- File for installing python dependencies
└── README.md
```

# Installation

### Creation of Virtual Environment
```bash
conda create --name <name> python=3.8.5 -y
conda activate <name>
```
### Install

```bash
pip install -r requirements.txt
```

# Usage

### train

```bash
python ./src/train.py
```

### How to change settings

By changing the settings in conf -> config.yaml,
Parameters can be changed in various places

# Note

The dataset is automatically downloaded the first time.

CPU and GPU switching is also automatically determined.

# License

"WavetableCVAE" is under [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.ja).






