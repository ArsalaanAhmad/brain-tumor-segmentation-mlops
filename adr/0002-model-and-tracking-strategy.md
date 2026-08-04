# ADR-0002: Model Architecture and Experiment Tracking

## Status: Accepted

## Context
Need a segmentation architecture for 3D multi-modal MRI (BraTS 2021),
uncertainty quantification for clinical trustworthiness, and a way to 
track and demonstrate training runs publicly in a portfolio context.

## Decisions

**Architecture:** 3D U-Net via MONAI.
Standard, well-validated architecture for volumetric medical image 
segmentation. MONAI's implementation is production-grade and used 
across academic and industry medical imaging pipelines.

**Input:** All 4 MRI modalities (T1, T1ce, T2, FLAIR).
Standard BraTS setup — each modality captures different tissue 
characteristics; using all 4 maximises available clinical signal.

**Uncertainty:** Monte Carlo Dropout.
Dropout layers remain active at inference time; running N forward 
passes produces a distribution of predictions whose variance 
constitutes an uncertainty map. Simple to implement, no additional 
training cost, well-cited in medical imaging literature.

**Export:** ONNX.
Universal, compute-backend-agnostic export format. Single 
torch.onnx.export call, no framework conversion overhead.

**Experiment tracking:** Weights & Biases (free tier).
Live dashboard with shareable run links — provides public, 
verifiable evidence of training runs directly from the README.

## Alternatives Considered
- Deep Ensembles for uncertainty: more robust but 3-5x compute cost, 
  overkill for portfolio scope.
- TFLite: more steps to convert from PyTorch vs ONNX, minimal 
  benefit for this use case.
- JSON metrics only: no shareable evidence of actual training runs.

## Consequences
W&B API key required at training time (set via environment variable, 
never hardcoded). ONNX export added as a post-training step in pipeline.