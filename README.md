---
title: QAT
emoji: 🚀
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
---

# Visual Wake Words with MobileViT-XXS

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/hello12w/QAT)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316?logo=gradio&logoColor=white)](https://www.gradio.app/)

A lightweight person-presence classifier built with **MobileViT-XXS**, trained for the
Visual Wake Words task, and served through an interactive Gradio interface. The model
uses quantization-aware training (QAT) and runs inference with ONNX Runtime.

> **Try it online:** [Open the live Hugging Face Space](https://huggingface.co/spaces/hello12w/QAT)

## Overview

Visual Wake Words is a binary image-classification task: determine whether an image
contains a person. This project provides a compact inference application designed for
edge-oriented experimentation.

The application:

- accepts live webcam frames or uploaded images;
- resizes and normalizes each image for MobileViT;
- runs inference with a quantized ONNX model; and
- displays confidence scores for **Person** and **Background**.

## Model details

| Property | Value |
| --- | --- |
| Architecture | MobileViT-XXS |
| Task | Binary person-presence classification |
| Dataset | COCO Visual Wake Words |
| Input | RGB image, resized to 224 × 224 |
| Classes | Person, Background |
| Training | Quantization-aware training |
| Export format | ONNX |
| Inference engine | ONNX Runtime |

During QAT, convolution layers use per-channel INT8 fake quantization while transformer
blocks remain in FP32. The included ONNX model is tracked with Git LFS.

## Repository structure

```text
MobileVit/
├── app.py                 # Gradio UI, preprocessing, and inference
├── mobilevit_qat.onnx     # Quantized MobileViT model (Git LFS)
├── requirements.txt       # Python dependencies
├── .gitattributes         # Git LFS tracking rules
└── README.md              # Project documentation
```

## Getting started

### Prerequisites

- Python 3
- Git
- [Git LFS](https://git-lfs.com/)

### Installation

1. Clone the repository and enter the project directory:

   ```bash
   git lfs install
   git clone https://github.com/101Vaibhav04/MobileVit.git
   cd MobileVit
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   On Windows:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Run locally

Start the Gradio application:

```bash
python app.py
```

Open the local URL printed in the terminal, then choose one of the two input modes:

- **Webcam** for streaming predictions from a camera.
- **Upload** for classifying a saved image.

## Inference pipeline

1. Convert the input to RGB.
2. Resize it to 224 × 224 pixels.
3. Scale pixel values to the `[0, 1]` range.
4. Normalize with ImageNet mean and standard deviation.
5. Convert the image from HWC to NCHW layout and add a batch dimension.
6. Run the ONNX model and apply softmax to obtain class probabilities.

## Notes and limitations

- This is a binary image classifier, not an object detector; it does not return bounding
  boxes or person counts.
- Predictions depend on the training distribution and should be evaluated before use in
  safety-critical or production systems.
- Ensure Git LFS has downloaded `mobilevit_qat.onnx` before launching the application.

## References

- [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)
- [Visual Wake Words dataset](https://www.tensorflow.org/datasets/catalog/visual_wake_words)
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
- [Gradio documentation](https://www.gradio.app/docs)

## Acknowledgements

This project builds on MobileViT, the COCO Visual Wake Words task, ONNX Runtime, and
Gradio.
