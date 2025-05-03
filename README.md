
# Fine-Grained Bird Species Classification on CUB-200 with ResNet-50

## Overview
This repository holds the deliverables for our project on fine-grained classification of 200 bird species using a ResNet-50 pipeline on the CUB-200-2011 dataset. We combine strong data augmentations, OneCycleLR scheduling, phased fine-tuning, label smoothing, MixUp, and CBAM attention to achieve **79.2 % Top-1**  accuracy with minimal overfitting.


## Files

- **final_report.pdf**  
  Detailed report covering:
  - Problem statement & motivation  
  - Related work & code references  
  - Methodology & model architecture  
  - Data preparation & preprocessing  
  - Quantitative results & error analysis  
  - Conclusions & future directions  

- **slide.pptx**  
  Presentation slides summarizing:
  - Objectives & approach  
  - Core techniques (augmentations, OneCycleLR, CBAM, MixUp)  
  - Main results
  - Conclusions & next steps  

## How to View

1. Open **final_report.pdf** for the full report.  
2. Open **slide.pptx** for the presentation summary.

## Acknowledgements

- **PyTorch & torchvision** for ResNet-50 implementation  
- **jongchan/attention-module** for CBAM code  
- **Facebook Research** for MixUp implementation  

**Authors:** Shiheng Xu · Renjie Fan · Kunshu Yang  
**Date:** May 2025  
```
