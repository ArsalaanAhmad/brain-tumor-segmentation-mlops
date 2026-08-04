# ADR-0003: BraTS Label Remapping Strategy

## Status: Accepted

## Context
BraTS 2021 segmentation masks use four label values: 0 (background), 
1 (necrotic tumour core), 2 (peritumoral oedema), and 4 (enhancing 
tumour). The label set is therefore non-contiguous — value 3 does not 
exist in the dataset.

MONAI's DiceLoss with to_onehot_y=True assumes contiguous integer 
labels starting from 0. With out_channels=4 (valid indices 0–3), 
encountering label value 4 causes an index out of bounds assertion 
on the CUDA kernel, crashing training immediately.

## Decision
Remap label values [0, 1, 2, 4] → [0, 1, 2, 3] using MONAI's 
MapLabelValued transform, applied before any loss computation. 
The remapping is inserted into the transform pipeline immediately 
after EnsureChannelFirstd, before normalisation or cropping.

## Alternatives Considered
- Set out_channels=5 to accommodate label 4: wastes a channel on 
  a class that doesn't exist (label 3), adds unnecessary model 
  parameters, and produces meaningless predictions for an empty class.
- Post-process predictions to remap back: adds inference complexity 
  with no benefit — the clinical meaning of the three tumour 
  sub-regions is preserved regardless of the integer label used 
  internally.

## Consequences
Label indices in model outputs correspond to remapped values 
(0=background, 1=necrotic core, 2=oedema, 3=enhancing tumour). 
Any downstream visualisation or reporting code must account for 
this mapping when referencing original BraTS label conventions.