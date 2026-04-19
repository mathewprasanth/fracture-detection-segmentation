import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def draw_yolo_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """
    Draws YOLO bounding boxes on image.

    image : numpy array (H, W, 3) in RGB
    boxes : numpy array of shape (N, 5) — [class, cx, cy, w, h] normalized
    """
    image = image.copy()
    h, w  = image.shape[:2]

    for box in boxes:
        if len(box) == 0:
            continue

        cls, cx, cy, bw, bh = box

        # Convert normalized coords to absolute pixels
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        label = f'Fracture {cls:.0f}'
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return image


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.4):
    """
    Overlays binary segmentation mask on image with transparency.

    image : numpy array (H, W, 3) in RGB
    mask  : numpy array (H, W) — binary 0/1
    alpha : transparency of overlay (0=invisible, 1=opaque)
    """
    image   = image.copy().astype(np.float32)
    overlay = image.copy()

    # Apply color to fracture pixels only
    overlay[mask == 1] = color

    # Blend original image with overlay
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    return result.astype(np.uint8)


def overlay_gradcam(image, heatmap, alpha=0.5):
    """
    Blends Grad-CAM heatmap onto original image.

    image   : numpy array (H, W, 3) in RGB
    heatmap : numpy array (H, W) — values 0-1
    alpha   : transparency of heatmap overlay
    """
    image = image.copy()

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Convert to 0-255 and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    result = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    return result


def visualize_pipeline(image, boxes=None, mask=None, heatmap=None, save_path=None):
    """
    Full pipeline visualization — shows all three outputs side by side.

    image    : numpy array (H, W, 3) RGB — original X-ray
    boxes    : YOLO bounding boxes or None
    mask     : U-Net binary mask or None
    heatmap  : Grad-CAM heatmap or None
    save_path: path to save the figure or None to just display
    """
    cols   = 1 + (boxes is not None) + (mask is not None) + (heatmap is not None)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))

    if cols == 1:
        axes = [axes]

    idx = 0

    # Original image
    axes[idx].imshow(image)
    axes[idx].set_title('Original X-ray')
    axes[idx].axis('off')
    idx += 1

    # YOLO boxes
    if boxes is not None:
        img_boxes = draw_yolo_boxes(image, boxes)
        axes[idx].imshow(img_boxes)
        axes[idx].set_title('YOLO Detection')
        axes[idx].axis('off')
        idx += 1

    # U-Net mask
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        img_mask = overlay_mask(image, mask)
        axes[idx].imshow(img_mask)
        axes[idx].set_title('U-Net Segmentation')
        axes[idx].axis('off')
        idx += 1

    # Grad-CAM heatmap
    if heatmap is not None:
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
        img_heatmap = overlay_gradcam(image, heatmap)
        axes[idx].imshow(img_heatmap)
        axes[idx].set_title('Grad-CAM')
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    else:
        plt.show()

    plt.close()