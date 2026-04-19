import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image

from models.yolo.predict    import load_model as load_yolo, predict_single
from models.unet.predict    import load_model as load_unet, predict_mask
from explainability.gradcam import GradCAM
from utils.transforms       import get_unet_transforms

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / 'outputs/pipeline'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline(image_path, yolo_model=None, unet_model=None, gradcam=None, save=True):
    """
    Full inference pipeline: YOLO detection → U-Net segmentation → Grad-CAM explainability.

    Input:
        image_path  : str or Path to X-ray image
        yolo_model  : loaded YOLO model (loaded once externally for efficiency)
        unet_model  : loaded U-Net model
        gradcam     : GradCAM instance
        save        : save 4-panel output image to outputs/pipeline/

    Output: dict with keys —
        original     : np.ndarray (H, W, 3) RGB — original X-ray
        boxes        : np.ndarray (N, 5) [cls, cx, cy, w, h] normalized
        annotated    : np.ndarray (H, W, 3) RGB — YOLO boxes drawn on image
        mask         : np.ndarray (H, W) uint8 binary — 1=fracture 0=background
        mask_overlay : np.ndarray (H, W, 3) RGB — red mask blended on original
        heatmap      : np.ndarray (H, W, 3) RGB — raw Grad-CAM colormap
        gradcam_out  : np.ndarray (H, W, 3) RGB — heatmap blended on original
        fracture_detected : bool
    """
    image_path = Path(image_path)

    # load models if not passed in
    if yolo_model is None:
        yolo_model = load_yolo()
    if unet_model is None:
        unet_model = load_unet()
        unet_model.eval()
    if gradcam is None:
        device = next(unet_model.parameters()).device
        gradcam = GradCAM(unet_model, device=str(device))

    # get device from model — avoids mismatch between module-level DEVICE and actual model device
    device = next(unet_model.parameters()).device

    # Step 1: YOLO detection
    yolo_result = predict_single(yolo_model, image_path, save=False)
    original    = yolo_result['image']      # (H, W, 3) RGB
    boxes       = yolo_result['boxes']      # (N, 5) normalized
    annotated   = yolo_result['annotated']  # image with boxes drawn
    h, w        = original.shape[:2]

    fracture_detected = len(boxes) > 0

    # Step 2: U-Net segmentation
    transform    = get_unet_transforms(train=False)
    augmented    = transform(image=original, mask=np.zeros((h, w), dtype=np.uint8))
    image_tensor = augmented['image'].unsqueeze(0).to(device)   # (1, 3, 512, 512)

    with torch.no_grad():
        output    = unet_model(image_tensor)             # (1, 1, 512, 512) raw logits
        mask_prob = torch.sigmoid(output)                # probabilities
        mask_bin  = (mask_prob > 0.3).squeeze().cpu().numpy().astype(np.uint8)

    # resize mask back to original image dimensions
    mask_resized = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    # red mask overlay
    mask_overlay = original.copy()
    mask_overlay[mask_resized == 1] = [255, 50, 50]
    mask_overlay = cv2.addWeighted(original, 0.6, mask_overlay, 0.4, 0)

    # Step 3: Grad-CAM
    heatmap         = gradcam.generate(image_tensor)              # (512, 512, 3)
    heatmap_resized = cv2.resize(heatmap, (w, h))                 # back to original size
    gradcam_overlay = gradcam.overlay(original, heatmap_resized)  # blended

    # Step 4: save 4-panel visualization
    if save:
        target_h = 512
        def resize_panel(img):
            scale = target_h / img.shape[0]
            return cv2.resize(img, (int(img.shape[1] * scale), target_h))

        panel = np.hstack([
            resize_panel(original),
            resize_panel(annotated),
            resize_panel(mask_overlay),
            resize_panel(gradcam_overlay)
        ])

        out_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f'Saved pipeline output: {out_path}')

    return {
        'original'         : original,
        'boxes'            : boxes,
        'annotated'        : annotated,
        'mask'             : mask_resized,
        'mask_overlay'     : mask_overlay,
        'heatmap'          : heatmap_resized,
        'gradcam_out'      : gradcam_overlay,
        'fracture_detected': fracture_detected
    }


if __name__ == '__main__':
    images_dir = ROOT / 'data/raw/FracAtlas/images'
    test_csv   = ROOT / 'data/splits/test.csv'
    df         = pd.read_csv(test_csv)

    # load models once — reuse across all images
    print('Loading models...')
    yolo_model = load_yolo()
    unet_model = load_unet()
    unet_model.eval()
    device     = next(unet_model.parameters()).device
    gradcam    = GradCAM(unet_model, device=str(device))
    print('Models loaded.\n')

    for image_name in df['image_id'].head(3):
        img_path = images_dir / 'Fractured' / image_name
        if not img_path.exists():
            img_path = images_dir / 'Non_fractured' / image_name

        result = run_pipeline(img_path, yolo_model, unet_model, gradcam, save=True)

        status = 'FRACTURE DETECTED' if result['fracture_detected'] else 'no fracture'
        print(f'{image_name}: {status} — {len(result["boxes"])} box(es)')
import numpy as np
import torch
import pandas as pd
from PIL import Image

from models.yolo.predict    import load_model as load_yolo, predict_single
from models.unet.predict    import load_model as load_unet, predict_mask
from explainability.gradcam import GradCAM
from utils.transforms       import get_unet_transforms

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / 'outputs/pipeline'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def run_pipeline(image_path, yolo_model=None, unet_model=None, gradcam=None, save=True):
    """
    Full inference pipeline: YOLO detection → U-Net segmentation → Grad-CAM explainability.

    Input:
        image_path  : str or Path to X-ray image
        yolo_model  : loaded YOLO model (loaded once externally for efficiency)
        unet_model  : loaded U-Net model
        gradcam     : GradCAM instance
        save        : save 4-panel output image to outputs/pipeline/

    Output: dict with keys —
        original     : np.ndarray (H, W, 3) RGB — original X-ray
        boxes        : np.ndarray (N, 5) [cls, cx, cy, w, h] normalized
        annotated    : np.ndarray (H, W, 3) RGB — YOLO boxes drawn on image
        mask         : np.ndarray (H, W) uint8 binary — 1=fracture 0=background
        mask_overlay : np.ndarray (H, W, 3) RGB — red mask blended on original
        heatmap      : np.ndarray (H, W, 3) RGB — raw Grad-CAM colormap
        gradcam_out  : np.ndarray (H, W, 3) RGB — heatmap blended on original
        fracture_detected : bool
    """
    image_path = Path(image_path)

    # ── load models if not passed in ───────────────────────
    if yolo_model is None:
        yolo_model = load_yolo()
    if unet_model is None:
        unet_model = load_unet()
        unet_model.eval()
    if gradcam is None:
        gradcam = GradCAM(unet_model, device=DEVICE)

    # ── Step 1: YOLO detection ──────────────────────────────
    yolo_result = predict_single(yolo_model, image_path, save=False)
    original    = yolo_result['image']      # (H, W, 3) RGB
    boxes       = yolo_result['boxes']      # (N, 5) normalized
    annotated   = yolo_result['annotated']  # image with boxes drawn
    h, w        = original.shape[:2]

    fracture_detected = len(boxes) > 0

    # ── Step 2: U-Net segmentation ──────────────────────────
    transform    = get_unet_transforms(train=False)
    augmented    = transform(image=original, mask=np.zeros((h, w), dtype=np.uint8))
    image_tensor = augmented['image'].unsqueeze(0).to(DEVICE)   # (1, 3, 512, 512)

    with torch.no_grad():
        output   = unet_model(image_tensor)             # (1, 1, 512, 512) raw logits
        mask_prob = torch.sigmoid(output)               # probabilities
        mask_bin  = (mask_prob > 0.3).squeeze().cpu().numpy().astype(np.uint8)

    # resize mask back to original image dimensions
    mask_resized = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    # red mask overlay
    mask_overlay = original.copy()
    mask_overlay[mask_resized == 1] = [255, 50, 50]
    mask_overlay = cv2.addWeighted(original, 0.6, mask_overlay, 0.4, 0)

    # ── Step 3: Grad-CAM ────────────────────────────────────
    heatmap         = gradcam.generate(image_tensor)              # (512, 512, 3)
    heatmap_resized = cv2.resize(heatmap, (w, h))                 # back to original size
    gradcam_overlay = gradcam.overlay(original, heatmap_resized)  # blended

    # ── Step 4: Save 4-panel visualization ─────────────────
    if save:
        # resize all panels to same height for hstack
        target_h = 512
        def resize_panel(img):
            scale = target_h / img.shape[0]
            return cv2.resize(img, (int(img.shape[1] * scale), target_h))

        panel = np.hstack([
            resize_panel(original),
            resize_panel(annotated),
            resize_panel(mask_overlay),
            resize_panel(gradcam_overlay)
        ])

        out_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f'Saved pipeline output: {out_path}')

    return {
        'original'        : original,
        'boxes'           : boxes,
        'annotated'       : annotated,
        'mask'            : mask_resized,
        'mask_overlay'    : mask_overlay,
        'heatmap'         : heatmap_resized,
        'gradcam_out'     : gradcam_overlay,
        'fracture_detected': fracture_detected
    }


if __name__ == '__main__':
    images_dir = ROOT / 'data/raw/FracAtlas/images'
    test_csv   = ROOT / 'data/splits/test.csv'
    df         = pd.read_csv(test_csv)

    # load models once — reuse across all images
    print('Loading models...')
    yolo_model = load_yolo()
    unet_model = load_unet()
    unet_model.eval()
    gradcam    = GradCAM(unet_model, device=DEVICE)
    print('Models loaded.\n')

    for image_name in df['image_id'].head(3):
        img_path = images_dir / 'Fractured' / image_name
        if not img_path.exists():
            img_path = images_dir / 'Non_fractured' / image_name

        result = run_pipeline(img_path, yolo_model, unet_model, gradcam, save=True)

        status = 'FRACTURE DETECTED' if result['fracture_detected'] else 'no fracture'
        print(f'{image_name}: {status} — {len(result["boxes"])} box(es)')