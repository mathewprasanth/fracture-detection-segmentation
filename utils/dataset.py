import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


class FracAtlasYOLODataset(Dataset):
    """
    Dataset for YOLO detection training.
    Reads images and their corresponding YOLO .txt annotation files.
    """

    def __init__(self, split_csv, images_dir, labels_dir, transforms=None):
        """
        split_csv   : path to train.csv / valid.csv / test.csv
        images_dir  : path to FracAtlas/images/
        labels_dir  : path to FracAtlas/Annotations/YOLO/
        transforms  : albumentations transform pipeline
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms

        # Load image filenames from split CSV
        df = pd.read_csv(split_csv)
        self.image_ids = df['image_id'].tolist()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx]

        # Load image — check Fractured first, then Non_fractured
        img_path = self.images_dir / 'Fractured' / image_name
        if not img_path.exists():
            img_path = self.images_dir / 'Non_fractured' / image_name

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load YOLO annotation
        label_name = Path(image_name).stem + '.txt'
        label_path = self.labels_dir / label_name
        boxes = []

        if label_path.exists():
            with open(label_path) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, w, h = map(float, parts)
                        boxes.append([cls, cx, cy, w, h])

        boxes = np.array(boxes, dtype=np.float32)

        if self.transforms:
            class_labels = boxes[:, 0].tolist() if len(boxes) > 0 else []
            bboxes = boxes[:, 1:].tolist() if len(boxes) > 0 else []
            transformed = self.transforms(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
            if len(bboxes) > 0:
                boxes = np.column_stack([class_labels, bboxes])
            else:
                boxes = np.array([], dtype=np.float32)

        return image, boxes


class FracAtlasUNetDataset(Dataset):
    """
    Dataset for U-Net segmentation training.
    Reads images and generates binary masks from COCO polygon annotations on the fly.
    """

    def __init__(self, split_csv, images_dir, coco_json_path, transforms=None):
        """
        split_csv      : path to train.csv / valid.csv / test.csv
        images_dir     : path to FracAtlas/images/
        coco_json_path : path to COCO_fracture_masks.json
        transforms     : albumentations transform pipeline
        """
        self.images_dir = Path(images_dir)
        self.transforms = transforms

        # Load split
        df = pd.read_csv(split_csv)
        self.image_ids = df['image_id'].tolist()

        # Load COCO JSON
        with open(coco_json_path) as f:
            coco = json.load(f)

        # Build lookup: filename -> image metadata
        self.filename_to_meta = {
            img['file_name']: img for img in coco['images']
        }

        # Build lookup: image_id -> list of annotations
        self.id_to_annotations = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.id_to_annotations:
                self.id_to_annotations[img_id] = []
            self.id_to_annotations[img_id].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def _generate_mask(self, image_name, orig_height, orig_width):
        """
        Convert COCO polygon annotations into a binary mask.
        White (1) = fracture, Black (0) = background.
        """
        mask = np.zeros((orig_height, orig_width), dtype=np.uint8)

        meta = self.filename_to_meta.get(image_name)
        if meta is None:
            # Non-fractured image — return empty mask
            return mask

        annotations = self.id_to_annotations.get(meta['id'], [])
        for ann in annotations:
            for polygon in ann['segmentation']:
                # polygon is [x1, y1, x2, y2, ...] — reshape to (N, 2)
                pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], color=1)

        return mask

    def __getitem__(self, idx):
        image_name = self.image_ids[idx]

        # Load image
        img_path = self.images_dir / 'Fractured' / image_name
        if not img_path.exists():
            img_path = self.images_dir / 'Non_fractured' / image_name

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_height, orig_width = image.shape[:2]

        # Generate binary mask from COCO polygons
        mask = self._generate_mask(image_name, orig_height, orig_width)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask