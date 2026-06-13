# Polyp Segmentation with ResUNet++

Deep learning model for automatic polyp segmentation in colonoscopy images, achieving **Dice Score: 0.8539** on CVC-ClinicDB.

## 🏆 Competition Background
Developed for TEKNOFEST 2025 AI in Health Competition (High School Category). Team **Galaktik İkizler** reached the international finals.

## 🧠 Model Architecture
**ResUNet++** with three key components:
- **Residual Blocks** — skip connections prevent vanishing gradients
- **Squeeze & Excitation** — channel-wise attention
- **ASPP** — multi-scale context aggregation

## 📊 Results
| Metric | Score |
|--------|-------|
| Dice Score | 0.8539 |
| IoU | 0.8780 |
| Accuracy | 0.9060 |
| Precision | 0.8690 |
| Recall | 0.8910 |
| F1 Score | 0.8800 |

## 📁 Dataset
**CVC-ClinicDB** — 612 colonoscopy frames  
Train: 489 | Validation: 123 | Size: 256×256

## 🚀 Quick Start
```bash
git clone https://github.com/zeynepceyhun152-code/polyp-segmentation
cd polyp-segmentation
pip install -r requirements.txt
python train.py
```

## 📂 Structure
polyp-segmentation/
├── train.py
├── requirements.txt
├── results/
│   ├── predictions.png
│   └── gradcam.png
└── README.md

## 👤 Author
**Zeynep** — TEKNOFEST 2025 Finalist  
Bilkent Erzurum Laboratory School (BELS)
