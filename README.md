# 🫁 Multi-Disease Chest X-Ray Detection

An AI system that detects 14 lung diseases from chest X-ray images using deep learning.  
Built with EfficientNetB4, custom TensorFlow training loop, Weighted Focal Loss, and Grad-CAM explainability.

🔗 **Live Demo:** [Coming soon — will be added after Hugging Face deployment]

---

## 🩺 Diseases Detected
Atelectasis · Cardiomegaly · Effusion · Infiltration · Mass · Nodule  
Pneumonia · Pneumothorax · Consolidation · Edema · Emphysema  
Fibrosis · Pleural Thickening · Hernia

---

## 🏗️ Project Architecture
Chest X-Ray Image
↓
tf.data Pipeline (load → augment → batch → prefetch)
↓
EfficientNetB4 (pretrained on ImageNet, fine-tuned)
↓
Weighted Focal Loss + Custom TF Training Loop
↓
14-class Sigmoid Output (multi-label)
↓
Grad-CAM Heatmap → Gradio Web App

---

## 📁 Project Structure
chestxray-project/
├── data/                      # CSV label files and split lists
├── notebooks/                 # Kaggle training notebooks
├── src/
│   ├── dataset.py             # tf.data pipeline with EfficientNet preprocessing
│   ├── losses.py              # Weighted Focal Loss implementation
│   ├── model.py               # EfficientNetB4 model builder
│   ├── train.py               # Custom training loop with tf.GradientTape
│   ├── evaluate.py            # AUC, ROC curves, comparison with CheXNet
│   └── gradcam.py             # Grad-CAM visualization
├── outputs/                   # Saved weights, plots, metrics
├── gradcam_results/           # Disease heatmap overlays
├── assets/                    # README images
├── app.py                     # Gradio web app
├── requirements.txt
└── README.md

---

## 📊 Results

| Disease | Your AUC | CheXNet AUC |
|---|---|---|
| Atelectasis | - | 0.8094 |
| Cardiomegaly | - | 0.9248 |
| Effusion | - | 0.8638 |
| Infiltration | - | 0.7345 |
| Mass | - | 0.8676 |
| Nodule | - | 0.7802 |
| Pneumonia | - | 0.7680 |
| Pneumothorax | - | 0.8887 |
| Consolidation | - | 0.7901 |
| Edema | - | 0.8878 |
| Emphysema | - | 0.9371 |
| Fibrosis | - | 0.8047 |
| Pleural Thickening | - | 0.8062 |
| Hernia | - | 0.9164 |
| **Mean AUC** | **0.72+** | **0.8414** |

> Results will be updated after training completes.  
> Baseline: CheXNet (DenseNet121) — Wang et al., Stanford 2017

---

## 🔬 Key Features

- **Custom TensorFlow training loop** using `tf.GradientTape`
- **Weighted Focal Loss** to handle severe class imbalance (Hernia: 0.2% prevalence)
- **tf.data pipeline** with EfficientNet-specific preprocessing, augmentation, and prefetching
- **Two-phase fine-tuning** — frozen base then unfreezing top 50 layers with cosine LR decay
- **Grad-CAM explainability** — highlights disease regions on X-ray
- **Gradio web app** — upload any chest X-ray and get instant predictions

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow 2.21, Keras |
| Model | EfficientNetB4 (ImageNet pretrained) |
| Data Pipeline | tf.data with EfficientNet preprocessing |
| Loss Function | Weighted Focal Loss (γ=2.0, α=0.25) |
| Training Strategy | Two-phase fine-tuning + Cosine LR Decay |
| Explainability | Grad-CAM |
| Web App | Gradio |
| Deployment | Hugging Face Spaces |
| Dataset | NIH ChestX-ray14 (112,120 images) |

---

## 🚀 How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/ayushsinha123/chestxray-disease-detection.git
cd chestxray-disease-detection
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add model weights**

Download `best_model_final.keras` from the releases section and place it in:
outputs/weights/best_model_final.keras

**5. Run the Gradio app**
```bash
python app.py
```

---

## 📓 Training

Training was done on Kaggle (2x Tesla T4 GPU).  
- Phase 1: 3 epochs, frozen EfficientNetB4 base, LR = 1e-3  
- Phase 2: 15 epochs, top 50 layers unfrozen, cosine LR decay from 1e-4  

Dataset: [NIH Chest X-rays on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

---

## 📚 References

- [CheXNet: Radiologist-Level Pneumonia Detection (Stanford)](https://arxiv.org/abs/1711.05225)
- [EfficientNet: Rethinking Model Scaling (Google)](https://arxiv.org/abs/1905.11946)
- [Focal Loss for Dense Object Detection (Lin et al.)](https://arxiv.org/abs/1708.02002)
- [Grad-CAM (Selvaraju et al.)](https://arxiv.org/abs/1610.02391)
- [NIH ChestX-ray14 Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)

---

## 👤 Author

**Ayush**  
B.Tech CSE — Machine Learning Workshop 2  
*Project built as part of deep learning coursework*

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**.  
It is not a certified medical device and should not be used for clinical diagnosis.