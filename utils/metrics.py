import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Measures overlap between predicted mask and ground truth mask.
    Lower is better. 0 = perfect overlap.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Flatten both tensors to 1D
        predictions = predictions.view(-1)
        targets     = targets.view(-1).float()

        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss for U-Net training.
    BCE handles pixel-level accuracy.
    Dice handles overlap — prevents model from ignoring small fracture regions.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce         = nn.BCEWithLogitsLoss()
        self.dice        = DiceLoss()

    def forward(self, predictions, targets):
        targets = targets.float()
        bce_loss  = self.bce(predictions, targets)

        # Apply sigmoid before dice — BCEWithLogitsLoss handles it internally but Dice needs probabilities
        predictions_sigmoid = torch.sigmoid(predictions)
        dice_loss = self.dice(predictions_sigmoid, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def iou_score(predictions, targets, threshold=0.5, smooth=1.0):
    """
    Intersection over Union for binary segmentation.
    Higher is better. 1.0 = perfect overlap.

    predictions : model output tensor (logits or probabilities)
    targets     : ground truth binary mask
    threshold   : cutoff to convert probabilities to binary (0 or 1)
    """
    # Convert logits to binary mask
    predictions = (torch.sigmoid(predictions) > threshold).float()
    targets     = targets.float()

    intersection = (predictions * targets).sum()
    union        = predictions.sum() + targets.sum() - intersection

    return (intersection + smooth) / (union + smooth)


def dice_score(predictions, targets, threshold=0.5, smooth=1.0):
    """
    Dice Coefficient for binary segmentation.
    Higher is better. 1.0 = perfect overlap.

    predictions : model output tensor (logits or probabilities)
    targets     : ground truth binary mask
    threshold   : cutoff to convert probabilities to binary (0 or 1)
    """
    predictions = (torch.sigmoid(predictions) > threshold).float()
    targets     = targets.float()

    intersection = (predictions * targets).sum()

    return (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)


def evaluate_unet(predictions, targets, threshold=0.5):
    """
    Runs all U-Net metrics in one call.
    Returns a dict of scores for logging.

    Usage in training loop:
        scores = evaluate_unet(predictions, masks)
        print(scores['iou'], scores['dice'])
    """
    return {
        'iou':  iou_score(predictions, targets, threshold).item(),
        'dice': dice_score(predictions, targets, threshold).item(),
    }