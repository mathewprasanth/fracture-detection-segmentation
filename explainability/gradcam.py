import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:                     
    """
    Grad-CAM for U-Net.

    Hooks into the ResNet34 encoder's final conv layer (layer4).
    During forward pass — saves feature maps (activations).
    During backward pass — saves gradients w.r.t. those activations.
    Combines both to produce a spatial heatmap showing what the model focused on.

    Why layer4:
        Early layers detect edges and textures — too generic.
        layer4 is the deepest encoder layer — most semantically meaningful.
        It captures high-level fracture patterns before decoding begins.
    """

    def __init__(self, model, device='mps'):
        self.model       = model
        self.device      = device
        self.gradients   = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """
        Register forward and backward hooks on encoder layer4.
        forward hook  — intercepts output during forward pass, saves feature maps
        backward hook — intercepts gradients during backward pass, saves them
        """
        target_layer = self.model.encoder.layer4[-1].conv2

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor):
        """
        Generate Grad-CAM heatmap as a colormap image.

        Input:
            image_tensor : Tensor (1, 3, H, W) normalized float32 on device

        Output:
            heatmap : np.ndarray (H, W, 3) uint8 RGB — JET colormap
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)

        # forward pass — compute prediction
        output = self.model(image_tensor)        # (1, 1, H, W) raw logits
        output = torch.sigmoid(output)           # convert to probability

        # backward pass — gradient of mean prediction w.r.t. layer4 activations
        self.model.zero_grad()
        output.mean().backward()

        # global average pool gradients across spatial dims → channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # weighted sum of activation maps → one spatial map
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)   # only keep positive activations — negative means suppressing prediction

        # resize cam to match input image resolution
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        cam  = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)

        # normalize to 0-255
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = (cam * 255).astype(np.uint8)

        # apply JET colormap — blue=low attention, red=high attention
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        return heatmap

    def overlay(self, original_image, heatmap, alpha=0.4):
        """
        Blend heatmap onto original image.

        Input:
            original_image : np.ndarray (H, W, 3) uint8 RGB
            heatmap        : np.ndarray (H, W, 3) uint8 RGB — same size as original
            alpha          : heatmap opacity (0=invisible, 1=fully opaque)

        Output:
            blended : np.ndarray (H, W, 3) uint8 RGB
        """
        # resize heatmap if dimensions don't match
        if original_image.shape[:2] != heatmap.shape[:2]:
            h, w    = original_image.shape[:2]
            heatmap = cv2.resize(heatmap, (w, h))

        original = original_image.astype(np.float32)
        heatmap  = heatmap.astype(np.float32)
        blended  = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
        return blended.astype(np.uint8)


if __name__ == '__main__':
    import pandas as pd
    from utils.transforms import get_unet_transforms
    from models.unet.predict import load_model

    ROOT       = Path(__file__).parent.parent
    images_dir = ROOT / 'data/raw/FracAtlas/images'
    test_csv   = ROOT / 'data/splits/test.csv'
    output_dir = ROOT / 'outputs/gradcam'
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    model   = load_model()
    model.eval()
    gradcam = GradCAM(model, device=device)

    transform = get_unet_transforms(train=False)
    df        = pd.read_csv(test_csv)

    print(f'Generating Grad-CAM for {min(3, len(df))} images...\n')

    for image_name in df['image_id'].head(3):
        img_path = images_dir / 'Fractured' / image_name
        if not img_path.exists():
            img_path = images_dir / 'Non_fractured' / image_name

        # load and preprocess image
        original = cv2.imread(str(img_path))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        augmented    = transform(image=original, mask=np.zeros(original.shape[:2], dtype=np.uint8))
        image_tensor = augmented['image'].unsqueeze(0).to(device)

        # generate heatmap and overlay
        heatmap = gradcam.generate(image_tensor)
        overlay = gradcam.overlay(original, heatmap)

        # save
        out_path = output_dir / image_name
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f'Saved: {out_path}')