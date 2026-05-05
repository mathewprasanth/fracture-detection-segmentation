# 🦴 Bone Fracture Detection & Segmentation

End-to-end medical imaging pipeline using deep learning to detect, localise, and segment bone fractures from X-ray images — with explainability via Grad-CAM.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/mathewprasanth/FractureDetectionSegmentation)
[![Model Weights](https://img.shields.io/badge/Weights-HuggingFace-blue)](https://huggingface.co/mathewprasanth/FractureDetectionSegmentationWeights)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-red)](https://pytorch.org)

---

## 📊 Results

| Metric | Value |
|---|---|
| Detection mAP50 | 0.479 |
| Detection Precision | 0.709 |
| Detection Recall | 0.462 |
| Segmentation Dice Score | 0.6224 |
| Dataset | FracAtlas — 4,083 radiologist-annotated X-rays |
| Note | Single-institution dataset — metrics reflect dataset difficulty |

---

## 🧠 What It Does

Bone fractures in X-rays can be subtle and easily missed under workload pressure. This system automates fracture detection and highlights the exact fracture region, assisting radiologists with faster and more consistent diagnosis.

The pipeline performs three tasks:
- Detects fracture location using YOLO object detection
- Segments the exact fracture boundary pixel-by-pixel using U-Net
- Generates Grad-CAM heatmaps showing model attention regions

This reflects real-world medical AI workflows where detection, localisation, and interpretability are all critical for clinical trust and adoption.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Detection | YOLO (Ultralytics, YOLOv3 from YOLO26m base) |
| Segmentation | U-Net with ResNet34 encoder (segmentation_models_pytorch) |
| Explainability | Grad-CAM (custom PyTorch hooks) |
| Backend | PyTorch |
| UI | Gradio |
| Hosting | Hugging Face Spaces |
| Model Storage | Hugging Face Model Hub |
| Image Processing | OpenCV, Albumentations |

---

## 🏗️ Pipeline Architecture

```
X-ray Image
→ YOLO Detection (bounding box + confidence)
→ U-Net Segmentation (pixel mask)
→ Grad-CAM (attention heatmap)
→ Output: detection + mask + explainability
```

---

## 🔑 Key Training Decisions

- 640px resolution outperformed 1280px due to dataset size constraints
- Low confidence threshold (0.05) to capture subtle fractures
- Combined BCE + Dice loss for stable segmentation training
- ReduceLROnPlateau scheduler for training stability
- Device-agnostic design (CPU / MPS / CUDA)

---

## 📁 Project Structure

```
fracture-detection-segmentation/
├── app.py
├── requirements.txt
├── configs/
├── data/
├── models/
├── explainability/
├── pipeline/
├── utils/
└── outputs/
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/mathewprasanth/fracture-detection-segmentation.git
cd fracture-detection-segmentation
pip install -r requirements.txt
python app.py
```

## Training

```bash
python models/yolo/train.py
python models/unet/train.py
```

---

## ⚠️ Limitations

- Single-institution dataset (FracAtlas) — generalisation to other hospital equipment not validated
- mAP50 of 0.479 reflects dataset difficulty (717 fractured vs 3,366 non-fractured) — larger multi-institution datasets would improve results significantly
- Binary fracture detection only — fracture type classification not yet supported

---

## 👤 Author

**Mathew Prasanth, P.E.**
AI/ML Engineer | U.S. Licensed Professional Engineer
[LinkedIn](https://www.linkedin.com/in/mathewprasanth/) · [Live Demo](https://huggingface.co/spaces/mathewprasanth/FractureDetectionSegmentation)

*AWS Certified ML Specialty · AWS Cloud Practitioner*
