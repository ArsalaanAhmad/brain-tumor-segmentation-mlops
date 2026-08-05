# ADR-0005: Uncertainty Quantification Strategy

## Status: Accepted

## Context
Clinical AI models require not just predictions but calibrated confidence 
estimates. A segmentation model that predicts a tumour boundary with low 
confidence should communicate that uncertainty to clinicians rather than 
presenting all predictions with equal authority. This is a regulatory 
concern (FDA/MHRA guidance on AI-assisted diagnosis increasingly expects 
uncertainty disclosure) and a clinical safety concern.

## Decision
Use Monte Carlo Dropout (MC Dropout) for uncertainty quantification.

During training, dropout layers (p=0.2) are already present in the U-Net 
architecture for regularisation. At inference time, rather than disabling 
dropout (standard eval() behaviour), we keep it active and run N=20 
forward passes on the same input. The variance across these 20 predictions 
constitutes a voxel-wise uncertainty map — high variance indicates regions 
where the model is uncertain about its segmentation boundary.

## Alternatives Considered
- **Deep Ensembles:** train 3-5 independent models, use variance across 
  them as uncertainty. More robust than MC Dropout empirically, but 
  requires 3-5x training compute and storage. Overkill for portfolio scope.
- **Temperature Scaling:** post-hoc calibration of softmax outputs. 
  Improves calibration but does not produce spatial uncertainty maps — 
  less clinically interpretable for segmentation tasks.
- **Evidential Deep Learning:** theoretically principled but requires 
  architectural changes and retraining. Not justified at this stage.

## Consequences
- No retraining required — dropout layers already present from training
- Inference time increases by ~20x (N=20 forward passes per sample) — 
  acceptable for offline clinical analysis, not suitable for real-time use
- Produces two outputs per inference: mean prediction (best estimate) 
  and variance map (uncertainty) — both saved and visualised
- Uncertainty maps are particularly meaningful at tumour boundaries, 
  which is clinically the most important region for surgical planning