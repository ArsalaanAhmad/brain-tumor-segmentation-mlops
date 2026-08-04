# ADR-0001: Dataset and Compute Strategy

## Status: Accepted

## Context
Need a medical imaging segmentation dataset for a pharma/clinical-AI portfolio 
project. No local GPU, cannot store large datasets on laptop.

## Decision
Use BraTS 2021 (brain tumor MRI segmentation) via Kaggle-hosted mirror. 
Training runs on Kaggle's free GPU tier by cloning this repo into a Kaggle 
notebook and invoking src/train.py directly — keeping all logic in versioned 
Python files, not notebook cells.

## Alternatives Considered
- LIDC-IDRI (lung nodules): weaker narrative tie to existing MONAI/saliency work.
- Local training: no GPU, dataset too large.

## Consequences
Pipeline must be compute-backend-agnostic — data paths live in config, 
not hardcoded in src/.