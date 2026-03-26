# Market-1501 Dataset Guide

## 📁 Dataset Structure

After extracting the dataset, you should have:

```
Market-1501-v15.09.15/
│
├── bounding_box_train/     # 12,936 images, 751 identities → TRAINING
├── bounding_box_test/      # 19,732 images, 750 identities → GALLERY  
├── query/                  # 3,368 images → PROBE (search queries)
├── gt_query/               # Ground truth (for evaluation only)
└── gt_bbox/                # Hand-drawn boxes (not used in training)
```

## 🏷️ Understanding the Filename Convention

Each image is named like: `0001_c1s1_001051_00.jpg`

| Part | Meaning | Example |
|------|---------|---------|
| `0001` | Person ID (identity) | Person #1 |
| `c1` | Camera ID | Camera 1 (of 6) |
| `s1` | Sequence number | Sequence 1 |
| `001051` | Frame number | Frame 1051 |
| `00` | Detection index | 1st detection in frame |

### Special Person IDs

| ID | Meaning | What to do |
|----|---------|------------|
| `0001` to `1501` | Valid person identities | ✅ Use for training |
| `0000` | **Distractors** (false detections) | ⚠️ Exclude from training, keep in gallery |
| `-1` | **Junk** images | ❌ Ignore completely |

## 🔄 Re-ID Training Protocol

### Training Set
- **Folder**: `bounding_box_train/`
- **Identities**: 751 unique people
- **Images**: 12,936
- **Use**: Train your Re-ID model

### Testing Protocol
1. **Query Set** (`query/`): 
   - 3,368 probe images
   - These are "search queries" - find this person in the gallery

2. **Gallery Set** (`bounding_box_test/`):
   - 19,732 images (including distractors)
   - This is the "database" to search through

### Evaluation Rules (IMPORTANT!)
For each query image:
1. Search through the entire gallery
2. **Exclude** gallery images with **same person ID AND same camera**
3. Rank remaining gallery images by similarity
4. Calculate CMC and mAP

## 📊 Standard Metrics

| Metric | Description | Good Score |
|--------|-------------|------------|
| **mAP** | Mean Average Precision | > 80% |
| **Rank-1** | Top-1 accuracy | > 90% |
| **Rank-5** | Top-5 accuracy | > 95% |
| **Rank-10** | Top-10 accuracy | > 97% |

## 🖼️ Image Characteristics

- **Size**: Variable (detected by DPM), typically ~128×256
- **Standard resize**: 256×128 (Height × Width)
- **Aspect ratio**: ~2:1 (portrait orientation)
- **Color space**: RGB

## 💻 Quick Start Code

```python
from src.data.market1501_dataset import (
    load_dataset_split,
    get_dataset_statistics,
    Market1501Dataset  # PyTorch Dataset
)

# Load training images
train_images = load_dataset_split("path/to/bounding_box_train")
print(f"Loaded {len(train_images)} training images")

# Get statistics
stats = get_dataset_statistics(train_images)
print(f"Identities: {stats['num_identities']}")
print(f"Cameras: {stats['camera_ids']}")

# PyTorch DataLoader
from torch.utils.data import DataLoader

dataset = Market1501Dataset(
    root="path/to/Market-1501-v15.09.15",
    split='train'
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## 🎯 Training Tips

### 1. Data Augmentation
```python
transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.ColorJitter(brightness=0.2, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5)  # Important for Re-ID!
])
```

### 2. Loss Functions
- **Triplet Loss**: Anchor + Positive (same ID) + Negative (diff ID)
- **Cross-Entropy + Center Loss**: Classification + feature clustering
- **Circle Loss**: Modern alternative to triplet loss

### 3. Sampling Strategy
Use **PK Sampling**: Each batch has P identities with K images each
- Example: P=8 identities × K=4 images = 32 batch size

## ⚠️ Common Mistakes

1. **Including junk images in training** → Filter out ID=-1
2. **Using same-camera images as positives** → Cross-camera positives are better
3. **Not excluding same-camera gallery for evaluation** → Inflates metrics
4. **Wrong image size** → Use 256×128 (H×W), not 128×256

## 📚 References

```bibtex
@inproceedings{zheng2015scalable,
  title={Scalable Person Re-identification: A Benchmark},
  author={Zheng, Liang and Shen, Liyue and Tian, Lu and Wang, Shengjin 
          and Wang, Jingdong and Tian, Qi},
  booktitle={ICCV},
  year={2015}
}
```
