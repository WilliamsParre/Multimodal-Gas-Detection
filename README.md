# Multimodal Gas Detection Benchmarking

> Fusing **gas sensor time-series** with **thermal imagery** to detect fire hazards earlier and more reliably than either modality can on its own.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-DeiT-blueviolet)](https://github.com/huggingface/pytorch-image-models)

---

## Overview

Most facilities exposed to fire risk, whether construction sites, warehouses, or industrial plants, instrument themselves with either a **gas sensor array** or a **thermal camera**, rarely both. The trouble is that each modality, in isolation, tells only half the story. Gas sensors return rich numerical readings about *what* is in the air, but say nothing about *where* it is spreading. Thermal cameras capture spatial heat signatures, but cannot identify the gas itself. In a real incident, that gap is exactly where early warning is lost.

This project addresses that gap. We benchmark a family of unimodal and multimodal deep learning models on the [MultimodalGasData](https://www.mdpi.com/2306-5729/7/8/112) dataset and show that fusing **LSTM-encoded sensor signals** with **DeiT-encoded thermal images** lifts test accuracy from the high-80s/low-90s (unimodal) to **100%** (multimodal), under both Early Fusion and Late Fusion strategies. Beyond the numbers, the goal was to set a clean, reproducible benchmark that other researchers in multimodal gas detection can build on.

Originally completed as the final research project for **CSCI 566 (Deep Learning and Its Applications)** at the University of Southern California by [Jash Prakash Shah](mailto:jashprak@usc.edu) and [Williams Parre](mailto:parre@usc.edu).

---

## Headline Results

| Model | Modality | Test Accuracy |
| --- | --- | :---: |
| LSTM | Gas sensors only | 0.9084 |
| DeiT (Feature Extraction) | Thermal images only | 0.8192 |
| DeiT (Fine-Tuning) | Thermal images only | 0.9901 |
| **Early Fusion (DeiT + LSTM)** | **Multimodal** | **1.0000** |
| **Late Fusion (DeiT + LSTM)** | **Multimodal** | **1.0000** |

The multimodal models clear both unimodal baselines, and importantly, the Late Fusion variant matches Early Fusion despite the larger accuracy gap between its underlying branches, validating the design choice of giving each branch its own trainable classification head rather than a naive probability average.

---

## Why Multimodal?

Past work in this domain (see [Narkhede et al., 2021](https://doi.org/10.3390/asi4010003)) reported 82% accuracy for LSTM on sensors and 93% for CNN on thermal images, with their multimodal fusion at 96%. More recent state-of-the-art such as [SCGA (Wang et al., 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0950423025002256) pushed unimodal methane detection to 99.2%. We hypothesized that the field's reliance on CNN-based image encoders was leaving accuracy on the table, since **transformers capture complex spatial relationships in images far more effectively than CNNs**. Swapping the image branch from CNN to DeiT, and pairing it with an LSTM on the sensor side, is the core architectural bet of this work.

---

## Architecture

The pipeline encodes the two modalities independently and then combines them via two distinct fusion strategies.

```
   Gas Sensor CSV (MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ135)
                       │
              ┌────────▼────────┐
              │  LSTM Encoder   │   (128 units, L2 reg, dropout 0.2)
              │   seq_len = 7   │
              └────────┬────────┘
                       │  numerical embeddings
                       │
                       │                  Thermal Image (224 × 224 × 3)
                       │                              │
                       │                     ┌────────▼────────┐
                       │                     │   DeiT Encoder  │
                       │                     │  (Fine-Tuned)   │
                       │                     │ 12 layers · 768d│
                       │                     └────────┬────────┘
                       │                              │  image embeddings
                       │                              │
                       └──────────┬──────────┬────────┘
                                  │          │
                       ┌──────────▼──┐   ┌───▼────────────┐
                       │ Early Fusion│   │  Late Fusion   │
                       │  Concat +   │   │  Per-branch    │
                       │  Dense Head │   │  Heads + Avg   │
                       └──────┬──────┘   └───────┬────────┘
                              │                  │
                              └────────┬─────────┘
                                       ▼
                          {No Gas, Perfume, Smoke, Mixture}
```

- **Early Fusion** concatenates `[h(image); h(num)]` at the feature level and passes the joint vector through a small dense head (128 units, GELU → Dropout → Softmax). Both unimodal backbones stay frozen; only the fusion head trains.
- **Late Fusion** keeps two independent classifier heads, one over DeiT features and one over LSTM features, and averages their softmax outputs. The trainable per-branch heads compensate for the accuracy imbalance between the strong DeiT branch (99.1%) and the weaker LSTM branch (91%), which a naive average would otherwise wash out.

---

## Dataset

We use the [MultimodalGasData](https://www.mdpi.com/2306-5729/7/8/112) dataset (Narkhede et al., 2022), consisting of:

- **6,400 sensor samples** in `Gas_Sensors_Measurements.csv`, with readings from seven MQ-series metal-oxide sensors (`MQ2`, `MQ3`, `MQ5`, `MQ6`, `MQ7`, `MQ8`, `MQ135`).
- **6,400 thermal images**, organized into per-class folders, captured in sync with the sensor readings.
- **4 classes**, 1,600 samples each: `NoGas`, `Perfume`, `Smoke`, `Mixture`.

The dataset is arranged sequentially by class, so splits are taken from the tail of each class to preserve temporal consistency (training 80%, validation 10%, testing 10% for the LSTM; 80/20 for the DeiT models).

> The dataset itself is not redistributed in this repository. Download it directly from the original authors and place `Gas_Sensors_Measurements.csv` and the thermal image folders at the paths referenced inside the notebooks.

---

## Repository Layout

```
.
├── Multimodal_Gas_Detection_Benchmarking.ipynb   # End-to-end notebook: LSTM, fusion training and eval
├── DeIT Model Fine-Tuning.ipynb                  # DeiT feature extraction + fine-tuning (PyTorch / timm)
├── Early Fusion for DL Project.ipynb             # Standalone walkthrough of the Early Fusion experiment
├── best_lstm_model.keras                         # Trained LSTM (Keras v3 format, Git LFS)
├── best_lstm_model.h5                            # Trained LSTM (legacy HDF5 format,  Git LFS)
├── Early-Fusion_DeIT_and_LSTM.keras              # Trained Early Fusion model (Git LFS)
├── Multimodal Gas Detection Benchmarking.pdf     # Full research paper
├── .gitattributes                                # Git LFS tracking rules (*.keras, *.h5, ipynb)
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python ≥ 3.10
- [Git LFS](https://git-lfs.com/) (required to pull the trained model weights)
- A CUDA-capable GPU is recommended for DeiT fine-tuning, but not strictly required.

### Setup

```bash
# 1. Clone with LFS so the .keras / .h5 weights come through
git lfs install
git clone <repo-url>
cd Multimodal-Gas-Detection

# 2. Create an isolated environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install tensorflow keras torch torchvision timm \
            pandas numpy scikit-learn matplotlib \
            jupyterlab
```

### Running the Notebooks

```bash
jupyter lab
```

Open the notebooks in the order they would typically be run:

1. **`Multimodal_Gas_Detection_Benchmarking.ipynb`** — trains the LSTM on sensor data, evaluates it, and serves as the main reproducibility entry point.
2. **`DeIT Model Fine-Tuning.ipynb`** — runs both DeiT variants (frozen feature extraction at lr = 3e-5, and full fine-tuning at lr = 3e-5, AdamW, CrossEntropyLoss, 7 epochs).
3. **`Early Fusion for DL Project.ipynb`** — extracts feature embeddings from the trained LSTM and fine-tuned DeiT, then trains the Early Fusion head (batch size 64, up to 50 epochs with `EarlyStopping` on `val_accuracy`).

The Late Fusion experiment is documented end-to-end inside the main benchmarking notebook.

### Using the Pretrained Models

```python
from tensorflow.keras.models import load_model

lstm = load_model("best_lstm_model.keras")
early_fusion = load_model("Early-Fusion_DeIT_and_LSTM.keras")
```

---

## Methodology, in Brief

### LSTM (Gas Sensors)

A three-layer sequential classifier (LSTM → Dropout → Dense-Softmax) trained on length-7 sliding windows over the seven MQ-sensor features. Initial training reached 88% test accuracy; loading those weights and continuing training (transfer learning on the same architecture) lifted the model to **90.8%**, which we adopted as the LSTM branch baseline.

### DeiT (Thermal Images)

We use `deit_base_patch16_224` from `timm`, pretrained on ImageNet. Two variants are benchmarked:

- **Feature Extraction.** The transformer encoder is frozen and only a fresh 4-class linear head is trained. Cheap, fast, and tops out at **82%**.
- **Fine-Tuning.** All 12 encoder layers are made trainable. The model fully adapts to thermal imagery and crosses **99%**.

The fine-tuned variant is the one fed into both fusion models, because its head start in accuracy is exactly what we want the fusion stage to amplify rather than fight.

### Fusion

- **Early Fusion** trains only the joint classification head over concatenated `[image, sensor]` embeddings. Both backbones stay frozen. This converges in 9 epochs (early-stopped) and reaches **100%** on the test set with a loss of 0.0234.
- **Late Fusion** trains one classifier head per branch and averages their softmax outputs. Stopped at epoch 5, also reaching **100%** test accuracy (loss 0.0570). The per-branch heads are the trick here. A naive averaging fusion would let the stronger DeiT branch dominate; giving each side a learnable head lets the model rebalance contributions.

---

## Discussion

A handful of observations worth highlighting:

- **Fine-tuning the DeiT branch matters disproportionately.** The 18-point jump from feature extraction (82%) to fine-tuning (99%) is the single largest accuracy delta in the project and is what makes the fusion stage easy.
- **Late Fusion is competitive with Early Fusion when the branches are given their own classifier heads.** This is the design lesson we'd most want to carry into future work.
- **The dataset's ceiling is in sight.** Both fusion models hit 100% test accuracy on MultimodalGasData; meaningful future progress will require a harder benchmark, perhaps with overlapping gas types or adversarial thermal conditions.
- **The LSTM is the weakest link.** A natural next step is to revisit the sensor branch, with a 1D CNN + BiGRU hybrid (in the spirit of [Wang et al., 2025](https://www.sciencedirect.com/science/article/abs/pii/S0950423025002256)) being an obvious candidate.

---

## Authors

- **Jash Prakash Shah** — `jashprak@usc.edu`
- **Williams Parre** — `parre@usc.edu`

University of Southern California · Viterbi School of Engineering

---

## Acknowledgments

Thanks to **Narkhede et al.** for publicly releasing the MultimodalGasData dataset, to **Facebook AI Research** for the DeiT architecture, and to the **CSCI 566** course staff at USC for the structure within which this project was carried out.
