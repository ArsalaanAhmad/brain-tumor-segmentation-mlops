# ADR-0004: Training Performance Optimisations

## Status: Accepted

## Context
Full BraTS 2021 dataset (1251 patients) with batch_size=1 and 3D volumes
at 128x128x64 was estimated to take 6-10 hours on a single T4 GPU — 
too slow for iterative development on Kaggle's free 12-hour session limit.
Three independent optimisations were identified that compound without 
affecting model architecture or clinical validity of outputs.

## Decisions

**Mixed Precision Training (torch.cuda.amp):**
Forward pass and loss computed in float16 rather than float32, with 
GradScaler managing gradient scaling to prevent underflow. Approximately 
2x speedup with no meaningful impact on convergence or final Dice score 
at this model scale. Standard practice in modern deep learning pipelines.

**DataLoader workers + pin_memory:**
num_workers=2 enables parallel data preprocessing on CPU while GPU 
runs the forward/backward pass — eliminates GPU starvation from 
sequential data loading. pin_memory=True accelerates CPU→GPU transfer 
by using page-locked memory.

**Reduced crop size (128x128x64 → 96x96x64):**
Smaller spatial crop reduces memory per volume, allowing faster 
iteration per batch. Tumour sub-regions in BraTS are spatially 
concentrated — a 96x96x64 crop retains sufficient context for 
segmentation while meaningfully reducing compute per step.

**Multi-GPU via DataParallel:**
If Kaggle provides >1 GPU (T4 x2), torch.nn.DataParallel distributes 
batches across both devices automatically. Conditional on 
torch.cuda.device_count() > 1 so pipeline degrades gracefully 
on single-GPU environments.

## Alternatives Considered
- Larger batch size: 3D volumes at this resolution already push T4 
  VRAM limits at batch_size=1 — increasing batch size risks OOM errors.
- torch.compile: available in PyTorch 2.x but adds compilation overhead 
  on first run and occasional compatibility issues with MONAI transforms; 
  deferred to a future optimisation pass.

## Consequences
ONNX export dummy input updated to match new roi_size (96x96x64). 
Mixed precision requires autocast context in both training and 
validation loops. Pipeline remains portable — all optimisations 
are hardware-conditional or config-driven.