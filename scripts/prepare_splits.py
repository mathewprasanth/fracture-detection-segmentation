"""
Adds non-fractured images to the official FracAtlas splits for YOLO training.
Run once from project root: uv run python scripts/prepare_splits.py
"""
from pathlib import Path
import pandas as pd
import random

random.seed(42)

nonfrac = list(Path('data/raw/FracAtlas/images/Non_fractured').glob('*.jpg'))
sample  = random.sample(nonfrac, 500)
names   = [p.name for p in sample]

train_neg = names[:400]
valid_neg = names[400:450]
test_neg  = names[450:]

for split, neg in [('train', train_neg), ('valid', valid_neg), ('test', test_neg)]:
    orig     = pd.read_csv(f'data/splits/{split}.csv')
    neg_df   = pd.DataFrame({'image_id': neg})
    combined = pd.concat([orig, neg_df], ignore_index=True)
    combined.to_csv(f'data/splits/{split}.csv', index=False)
    print(f'{split}: {len(combined)} total')