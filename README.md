# Brain Tumor Segmentation MLOps Pipeline

End-to-end MLOps pipeline for brain tumor segmentation on BraTS 2021 MRI data.
Built with MONAI, DVC, and a compute-agnostic design (runs on Kaggle, Colab, 
or local GPU without code changes).

## Features
- 3D U-Net segmentation (T1/T1ce/T2/FLAIR multi-modal MRI)
- Uncertainty quantification via Monte Carlo dropout
- Saliency overlays for model explainability
- Quantized model export (ONNX/TFLite) for edge deployment
- DVC-tracked pipeline with versioned configs and metrics
- Architecture decisions documented in /adr

## Stack
MONAI · PyTorch · DVC · ONNX · Python

## Status
🚧 In progress