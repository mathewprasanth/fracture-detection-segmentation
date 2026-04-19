

# Bone Fracture Detection and Segmentation

End-to-end medical imaging pipeline using deep learning to detect, localize, and segment bone fractures from X-ray images — with explainability via Grad-CAM.

**Live Demo** → https://huggingface.co/spaces/mathewprasanth/FractureDetectionSegmentation  
**Model Weights** → https://huggingface.co/mathewprasanth/FractureDetectionSegmentationWeights

---

## What It Does

Bone fractures in X-rays can be subtle and easily missed, especially under workload pressure. This system automates fracture detection and highlights the exact fracture region, assisting radiologists with faster and more consistent diagnosis.

The pipeline performs three tasks:
- Detects fracture location using object detection
- Segments the exact fracture boundary pixel-by-pixel
- Generates explainability heatmaps showing model attention

This reflects real-world medical AI workflows, where detection, localization, and interpretability are all critical for clinical trust and adoption.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Detection Model | YOLO (Ultralytics, YOLOv3 trained from YOLO26m base) |
| Segmentation Model | U-Net with ResNet34 encoder (segmentation_models_pytorch) |
| Explainability | Grad-CAM (custom implementation with PyTorch hooks) |
| Backend | PyTorch |
| UI | Gradio |
| Hosting | Hugging Face Spaces |
| Model Storage | Hugging Face Model Hub |
| Image Processing | OpenCV, NumPy |
| Data Pipeline | Albumentations |

---

## Dataset

**FracAtlas X-ray Dataset** — 4,083 labeled medical images.

| Category | Count |
|---|---|
| Fractured | 717 |
| Non-Fractured | ~3,366 |

- Binary classification + segmentation labels  
- Train/Validation/Test split via CSV files  
- Class imbalance handled during training  

---

## Pipeline Architecture

```

X-ray Image
→ YOLO Detection (bounding box + confidence)
→ U-Net Segmentation (pixel mask)
→ Grad-CAM (attention heatmap)
→ Output: detection + mask + explainability

```

---

## Model Architecture

### Detection (YOLO)
- mAP50: 0.479  
- Precision: 0.709  
- Recall: 0.462  

### Segmentation (U-Net + ResNet34)
- Loss: BCE + Dice Loss  
- Dice Score: 0.6224  

---

## Key Training Decisions

- 640px resolution outperformed 1280px due to dataset size constraints  
- Low confidence threshold (0.05) to capture subtle fractures  
- Combined BCE + Dice loss for stable segmentation  
- ReduceLROnPlateau scheduler for training stability  
- Device-agnostic design (CPU / MPS / CUDA)

---

## Project Structure

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

````

---

## Inference Pipeline

1. Upload X-ray image  
2. YOLO detects fracture location  
3. U-Net segments fracture region  
4. Grad-CAM generates attention heatmap  
5. Output includes detection, mask, and explainability  

---

## Deployment

- Hosted on Hugging Face Spaces  
- Model weights stored on Hugging Face Model Hub  
- Downloaded dynamically at runtime  
- Runs on CPU (HF Spaces constraint)

---

## Key Engineering Decisions

- Device resolved dynamically from model parameters  
- Models loaded once at startup for performance  
- Clear separation between UI and inference pipeline  

---

## Run Locally

```bash
git clone https://github.com/mathewprasanth3/fracture-detection-segmentation.git
cd fracture-detection-segmentation
pip install -r requirements.txt
python app.py
````

---

## Training

```bash
python models/yolo/train.py
python models/unet/train.py
```

---

## Results

| Metric     | Value  |
| ---------- | ------ |
| mAP50      | 0.479  |
| Precision  | 0.709  |
| Recall     | 0.462  |
| Dice Score | 0.6224 |

---

## Author

**Mathew Prasanth, PE**
AI/ML Engineer
[https://www.linkedin.com/in/mathewprasanth/](https://www.linkedin.com/in/mathewprasanth/)
[https://huggingface.co/spaces/mathewprasanth/FractureDetectionSegmentation](https://huggingface.co/spaces/mathewprasanth/FractureDetectionSegmentation)

AWS Certified Cloud Practitioner · AWS Certified Machine Learning Specialty



