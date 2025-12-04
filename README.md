# Brain Tumor Detection Using Hybrid Deep Learning Models with Explainable AI

A comprehensive deep learning project implementing hybrid architectures for accurate brain tumor detection and classification from MRI images, with explainable AI techniques for clinical interpretability.

## 🎯 Project Overview

This project explores multiple approaches to brain tumor detection:
- **YOLOv10-based object detection** for tumor localization
- **Two novel hybrid deep learning architectures** for tumor classification
- **Explainable AI (Grad-CAM)** for model interpretability

## 📊 Model Performance

| Model | Architecture | Test Accuracy | Key Innovation |
|-------|-------------|---------------|----------------|
| **Hybrid Model 1** | VGG19 + LSTM | **98%** | Sequential spatial feature learning |
| **Hybrid Model 2** | VGG16 + InceptionV3 | **97%** | Multi-architecture feature fusion |
| **YOLOv10** | Object Detection | **96.4% mAP@50** | Real-time tumor localization |

## 🏗️ Architecture Details

### Model 1: VGG19 + LSTM Hybrid

**Architecture Components:**
- **Feature Extractor:** Pre-trained VGG19 (ImageNet weights, frozen layers)
- **Sequential Processing:** LSTM layer (512 units) for capturing spatial dependencies
- **Classification Head:** Two fully connected layers (4096 units each) with BatchNormalization
- **Output:** 4-class softmax for tumor type classification

**Key Features:**
- Reshapes VGG19 output (7×7×512) → sequence format (49, 512) for LSTM processing
- LSTM captures temporal/spatial relationships in feature maps
- Heavy regularization with BatchNormalization
- Achieved **98% accuracy** on test set

**Innovation:** Combines CNN's spatial feature extraction with LSTM's ability to model sequential dependencies, treating the spatial feature map as a sequence.

### Model 2: VGG16 + InceptionV3 Ensemble

**Architecture Components:**
- **Dual Feature Extractors:** VGG16 and InceptionV3 (both pre-trained, frozen)
- **Feature Fusion:** Concatenation layer combining features from both branches
- **Dense Layers:** 1024-unit layers with strong regularization
- **Output:** 4-class softmax classifier

**Key Features:**
- Ensemble approach leveraging complementary architectures
- Global Average Pooling for dimensionality reduction
- Comprehensive regularization: L1-L2 + Dropout (0.7) + BatchNorm
- K-Fold Cross-Validation (5 folds) for robust evaluation
- Achieved **97% accuracy** on test set

**Innovation:** Combines VGG16's deep hierarchical features with InceptionV3's multi-scale processing for robust tumor classification.

## 📈 Comparative Analysis

| Aspect | VGG19+LSTM | VGG16+InceptionV3 |
|--------|------------|-------------------|
| **Approach** | Sequential processing | Feature ensemble |
| **Complexity** | Higher (LSTM + large FC) | Moderate (dual CNNs) |
| **Parameters** | Very high (4096×2 FC) | Lower (1024 dense) |
| **Regularization** | BatchNorm only | L1-L2 + Dropout + BatchNorm |
| **Validation** | Train/Val split | K-Fold CV |
| **Best For** | Capturing spatial context | Multi-scale feature fusion |
| **Accuracy** | **98%** | **97%** |

## 🧠 Tumor Classification

The models classify brain tumors into **4 categories:**
1. **Glioma** - Most common primary brain tumor
2. **Meningioma** - Tumor arising from meninges
3. **Pituitary** - Tumor in pituitary gland
4. **No Tumor** - Normal brain tissue

## 🔍 Explainable AI with Grad-CAM & LRP

Both models implement **explainable AI techniques** to visualize which regions of the MRI scan influenced the model's decision:
- **Grad-CAM (Gradient-weighted Class Activation Mapping)** - Highlights important regions
- **LRP (Layer-wise Relevance Propagation)** - Provides pixel-level contribution analysis

**Benefits:**
- Provides transparency in model predictions
- Helps clinicians understand AI decision-making
- Validates that models focus on relevant anatomical regions
- Builds trust in AI-assisted diagnosis

### Sample Tumor Detection Results

![image](https://github.com/user-attachments/assets/27d26f45-7e4d-4901-9a60-50f8732ecc02)
![image](https://github.com/user-attachments/assets/f54a4f21-dd31-4b0f-a172-6490299b1bce)
![image](https://github.com/user-attachments/assets/4b2351be-dfac-4644-b50f-2123e3a4e824)
![image](https://github.com/user-attachments/assets/9fac58ad-72a6-411e-bd35-92bbd2d92c15)

### Grad-CAM Visualizations

Grad-CAM heatmaps show which brain regions the model focused on for classification:

![image](https://github.com/user-attachments/assets/350409f8-c6ac-4491-bbb7-de5679133cc7)
![image](https://github.com/user-attachments/assets/ca74774c-2bba-4363-81e1-c006736ed7cc)
![image](https://github.com/user-attachments/assets/6b4572f6-cfc0-4ddf-9f4e-03cc77046683)
![image](https://github.com/user-attachments/assets/33c2f328-b2e2-46d0-83c7-9a569483364f)

### Layer-wise Relevance Propagation (LRP)

LRP provides pixel-level attribution showing exact contributions to predictions:

![image](https://github.com/user-attachments/assets/e07fc9bc-849a-4ad1-bfc0-296e7897f0e8)
![image](https://github.com/user-attachments/assets/bc8340b6-0c22-4f8a-8bb5-b06a4b68d1eb)
![image](https://github.com/user-attachments/assets/4f727083-7f46-462d-ab50-a08aacd72b52)
![image](https://github.com/user-attachments/assets/e7d14767-5f23-4e53-8eae-93a8657d4a94)

## 🛠️ Technologies Used

- **Deep Learning Frameworks:** TensorFlow, Keras, PyTorch
- **Computer Vision:** OpenCV, Ultralytics (YOLOv10)
- **Model Architectures:** VGG16, VGG19, InceptionV3, YOLOv10, LSTM
- **Explainable AI:** Grad-CAM
- **Data Processing:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Seaborn

## 📁 Project Structure

```
Brain-Tumor-Detection/
├── notebooks/
│   ├── yolov10_detection.ipynb          # YOLOv10 object detection
│   ├── vgg19_lstm_hybrid.ipynb          # Model 1: VGG19+LSTM
│   ├── vgg16_inceptionv3_ensemble.ipynb # Model 2: VGG16+InceptionV3
│   └── gradcam_visualization.ipynb      # Explainable AI
├── models/
│   ├── vgg19_lstm_best.h5
│   ├── vgg16_inception_best.h5
│   └── yolov10_tumor.pt
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── visualizations/
│   └── gradcam_outputs/
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.8+
PyTorch 1.12+
CUDA 11.7+ (for GPU acceleration)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/07fahim/Brain-Tumor-detection-Hybrid-Model-with-XAI-.git
cd Brain-Tumor-detection-Hybrid-Model-with-XAI-

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
tensorflow>=2.8.0
keras>=2.8.0
torch>=1.12.0
torchvision>=0.13.0
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 💻 Usage

### Training Models

**VGG19 + LSTM Hybrid:**
```bash
jupyter notebook notebooks/vgg19_lstm_hybrid.ipynb
```

**VGG16 + InceptionV3 Ensemble:**
```bash
jupyter notebook notebooks/vgg16_inceptionv3_ensemble.ipynb
```

**YOLOv10 Detection:**
```bash
jupyter notebook notebooks/yolov10_detection.ipynb
```

### Generating Grad-CAM Visualizations

```bash
jupyter notebook notebooks/gradcam_visualization.ipynb
```

## 📊 Results Summary

### Classification Performance

| Metric | VGG19+LSTM | VGG16+InceptionV3 |
|--------|------------|-------------------|
| **Accuracy** | 98% | 97% |
| **Precision** | 97.8% | 96.9% |
| **Recall** | 98.1% | 97.2% |
| **F1-Score** | 97.9% | 97.0% |

### YOLOv10 Detection Performance

| Model Variant | mAP@50 | mAP@50-95 | Inference Speed |
|---------------|---------|-----------|-----------------|
| YOLOv10-S | 96.4% | 78.2% | 8.5ms |
| YOLOv10-M | 96.8% | 79.1% | 12.3ms |
| YOLOv10-N | 95.2% | 76.5% | 6.2ms |
| YOLOv10-B | 97.1% | 80.3% | 15.7ms |

## 🔬 Key Findings

1. **VGG19+LSTM achieved the highest accuracy (98%)** by effectively capturing spatial dependencies in MRI scans
2. **Ensemble approach (97%) provided robust generalization** through complementary feature extraction
3. **Grad-CAM successfully highlighted tumor regions**, validating model decisions
4. **YOLOv10 enabled real-time detection** with 96.4% mAP@50 for clinical applications
5. **K-Fold CV in Model 2 ensured consistent performance** across different data splits

## 🎓 Research Publications

This work has been accepted and published at:

1. **IEEE RAAICON 2024** - "Deep Learning for Brain Tumor Detection Leveraging YOLOv10 for Precise Localization"
2. **ICCIT 2024** - "A Hybrid Deep Learning Approach For Brain Tumor Detection Using XAI with Grad-CAM"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Gazi Mohammad Fahim Faiyaz**
- GitHub: [@07fahim](https://github.com/07fahim)
- LinkedIn: [fahim-faiyaz](https://linkedin.com/in/fahim-faiyaz)
- Email: faiyazfahim743@gmail.com

## 🙏 Acknowledgments

- Dataset providers and medical imaging community
- TensorFlow and PyTorch teams for excellent frameworks
- Research community for pre-trained models (VGG, InceptionV3)
- Ultralytics for YOLOv10 implementation

## 📚 References

- VGG Networks: Simonyan & Zisserman (2014)
- InceptionV3: Szegedy et al. (2015)
- LSTM: Hochreiter & Schmidhuber (1997)
- Grad-CAM: Selvaraju et al. (2017)
- YOLOv10: Wang et al. (2024)

---

⭐ **If you find this project useful, please consider giving it a star!** ⭐

## 📞 Contact

For questions or collaborations, feel free to reach out through GitHub issues or email.
