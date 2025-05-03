
# Fine-Grained Bird Species Classification on CUB-200 with ResNet-50

## 📋 Project Overview
We tackle the fine-grained classification of 200 bird species in the CUB-200-2011 dataset by designing an efficient, single-model pipeline based on ResNet-50. By systematically integrating:

- **End-to-end**: `542.py` handles data loading, training and evaluation 
- **Key techniques**: strong augmentations, OneCycleLR scheduling, label smoothing, gradient clipping, CBAM attention  
- **Performance**: achieves **79.2 % Top-1**  with stable convergence and minimal overfitting  

we achieve **79.2 % ± 0.1 % Top-1**  accuracy, with validation accuracy stabilizing above 77 % by epoch 20 and only minimal overfitting.

## 📁 Repository Contents
```

├── README.md           ← this document
├── final\_report.pdf    ← full write-up (motivation, methodology, results, conclusion)
├── slide.pptx          ← project presentation slides
├── assets/
│   └── videos/
│       └── demo.mp4    ← short demo of the trained model in action
├── midpoint/           ← code & experiments up to midpoint report
└── final/              ← polished scripts, logs, and model weights for final evaluation

````

## 📄 How to Explore

1. **Read the report**  
   Open `final_report.pdf` for detailed descriptions of:
   - Problem statement & refined objectives  
   - Related work & open-source references  
   - Dataset preparation & preprocessing  
   - Model architecture & training regimen  
   - Quantitative results, learning curves & error analysis  
   - Conclusions & future work  

2. **View the slides**  
   Open `slide.pptx` for a concise summary of our approach and key findings.

3. **Inspect code and experiments**  
   - `midpoint/` branch: scripts and notebooks used in the project midpoint.  
   - `final/` branch: final training scripts and weights matching the report’s results.  

   ```bash
   # Switch between branches:
   git checkout midpoint   # midpoint experiments
   git checkout final      # final implementation

## 📚 Key References

* Wah et al., *The Caltech–UCSD Birds-200-2011 Dataset*, 2011.
* He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
* Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018.
* Zhang et al., *mixup: Beyond Empirical Risk Minimization*, ICLR 2018.
* Zhong et al., *Random Erasing Data Augmentation*, AAAI 2020.
* Smith & Topin, *Super-Convergence*, 2019.

*(See `final_report.pdf` bibliography for full citation details.)*

## 🤝 Acknowledgements

* **PyTorch & torchvision** for ResNet-50 implementation
* **jongchan/attention-module** for CBAM reference code
* **Facebook Research** for the original MixUp implementation

---

**Authors:** Shiheng Xu · Renjie Fan · Kunshu Yang
**Date:** May 2025


