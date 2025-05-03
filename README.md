
# Fine-Grained Bird Species Classification on CUB-200 with ResNet-50

## Overview
This repository holds the deliverables for our project on fine-grained classification of 200 bird species using a ResNet-50 pipeline on the CUB-200-2011 dataset. We combine strong data augmentations, OneCycleLR scheduling, phased fine-tuning, label smoothing, MixUp, and CBAM attention to achieve 79.2 % Top-1 accuracy with minimal overfitting.

## Repository Structure

├── README.md
├── final\_report.pdf    # Full write-up: motivation, methodology, results, conclusion
├── slide.pptx          # Slide deck: overview, key results, next steps
│
├── branches/
│   ├── midpoint        # Code and experiments up through the midpoint report
│   └── final           # Final code and experiments matching the final report


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
  - Main results (Top-1/Top-5, learning curves)  
  - Error-mode visualizations  
  - Conclusions & next steps  

## How to View

1. **Checkout the midpoint branch** for the code & experiments used in our midpoint report:

   git checkout midpoint

2. **Switch to the final branch** for the final implementation and trained models:
 
   git checkout final
   
3. Open **final\_report.pdf** for the full write-up.
4. Open **slide.pptx** for the presentation summary.


## Key References

* Wah et al., “The Caltech–UCSD Birds-200-2011 Dataset,” 2011.
* He et al., “Deep Residual Learning for Image Recognition,” CVPR 2016.
* Woo et al., “CBAM: Convolutional Block Attention Module,” ECCV 2018.
* Zhang et al., “mixup: Beyond Empirical Risk Minimization,” ICLR 2018.
* Zhong et al., “Random Erasing Data Augmentation,” AAAI 2020.
* Smith & Topin, “Super-Convergence,” 2019.

## Acknowledgements

* **PyTorch & torchvision** for ResNet-50 implementation
* **jongchan/attention-module** for CBAM code
* **Facebook Research** for MixUp implementation

**Authors:** Shiheng Xu · Renjie Fan · Kunshu Yang
**Date:** May 2025

```
