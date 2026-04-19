"""
app/app.py

Bone Fracture Detection and Segmentation Gradio demo.
Accepts an X-ray image, runs YOLO26m for fracture detection,
U-Net for pixel-level segmentation, and Grad-CAM for explainability.

Usage:
    uv run python app/app.py
"""

import sys
import traceback
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import gradio as gr
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from models.yolo.predict    import load_model as load_yolo
from models.unet.predict    import load_model as load_unet
from explainability.gradcam import GradCAM
from pipeline.inference     import run_pipeline

# download weights from HF model hub
REPO_ID = "mathewprasanth/FractureDetectionSegmentationWeights"

print("Downloading weights from Hugging Face...")
yolo_weights_path = hf_hub_download(repo_id=REPO_ID, filename="yolo_best.pt")
unet_weights_path = hf_hub_download(repo_id=REPO_ID, filename="unet_best.pth")

# load models once at startup
print('Loading models...')
yolo_model = load_yolo(weights_path=Path(yolo_weights_path))
unet_model = load_unet(weights_path=Path(unet_weights_path))
unet_model.eval()
device = next(unet_model.parameters()).device
gradcam = GradCAM(unet_model, device=str(device))
print('Models ready.')


def predict(image: np.ndarray):
    if image is None:
        return 'please upload an X-ray image', None, None, None, None

    tmp_path = Path('/tmp/input_xray.jpg')
    Image.fromarray(image).save(tmp_path)

    try:
        result = run_pipeline(
            image_path = tmp_path,
            yolo_model = yolo_model,
            unet_model = unet_model,
            gradcam    = gradcam,
            save       = False
        )
    except Exception as e:
        traceback.print_exc()
        return f'error: {str(e)}', None, None, None, None

    n = len(result['boxes'])
    if n == 0:
        label = 'No fracture detected'
    elif n == 1:
        label = '1 fracture detected'
    else:
        label = f'{n} fractures detected'

    return (
        label,
        result['original'],
        result['annotated'],
        result['mask_overlay'],
        result['gradcam_out'],
    )


with gr.Blocks(title='Bone Fracture Detection') as demo:

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown('# 🦴 Bone Fracture Detection & Segmentation')
            gr.Markdown(
                'End-to-end medical imaging pipeline for detecting and segmenting '
                'bone fractures in X-ray images. Upload an X-ray to analyze it.'
            )
        with gr.Column(scale=1):
            gr.Markdown(
                "<div style='text-align: right;'>"
                "<strong>Mathew Prasanth</strong><br>AI/ML Engineer"
                "</div>"
            )

    gr.Markdown(
        '> **Pipeline:** YOLO26m detects fracture location → '
        'U-Net segments exact boundary pixel by pixel → '
        'Grad-CAM shows model focus areas.'
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label  = 'Upload X-ray',
                type   = 'numpy',
                height = 400
            )
            analyze_btn = gr.Button('Analyze X-ray', variant='primary')

        with gr.Column(scale=2):
            result_label = gr.Textbox(
                label       = 'Result',
                lines       = 3,
                interactive = False
            )

    gr.Markdown('### Detection')

    with gr.Row():
        out_original  = gr.Image(label='Original X-ray',  height=350)
        out_detection = gr.Image(label='YOLO Detection',   height=350)

    gr.Markdown('### Segmentation')

    with gr.Row():
        out_segmentation = gr.Image(label='U-Net Segmentation', height=350)

    gr.Markdown('### Grad-CAM — where the model focused')
    gr.Markdown('Red/yellow = model focused here. Blue = model ignored.')

    with gr.Row():
        out_gradcam = gr.Image(label='Grad-CAM Heatmap', height=350)

    analyze_btn.click(
    fn      = predict,
    inputs  = [input_image],
    outputs = [result_label, out_original, out_detection, out_segmentation, out_gradcam]
)

    gr.Markdown('---')
    gr.Markdown(
        '**Model:** YOLO26m pretrained on COCO, fine-tuned on FracAtlas — '
        '4,083 radiologist-annotated X-rays. '
        'mAP50 0.479 | Dice 0.6224'
    )


demo.launch(
    server_name = '0.0.0.0',
    server_port = 7860,
)